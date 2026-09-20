#include "paged_attention/api.cuh"
#include <cmath>
namespace pa {

namespace {
__device__ float warp_reduce_sum(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}
__global__ void paged_attention_gqa_reuse_kernel(Problem p, Inputs in, float *out) {
  const int kv_head = blockIdx.x;
  const int batch = blockIdx.y;
  constexpr int vec_width = 4;
  constexpr int warp_size = 32;
  const int warp = threadIdx.x / warp_size;
  const int lane = threadIdx.x % warp_size;
  // since we are vectorizing, we will use chunks instead of head_dim
  const int chunks = p.head_dim / vec_width;
  bool is_valid_lane = lane < chunks;
  // each CTA contains four warps (handles 4 kv_heads)
  // each CTA handles 1 kv_head and (queries_per_kv) query_heads
  // each warp handles 1 query_head

  // all shared_k and shared_v for one page.
  extern __shared__ float4 shared[];
  float4 *shared_k = shared;
  float4 *shared_v = shared + p.page_size * chunks;
  // every page hosts (page_size * chunks) chunks
  const int query_head = kv_head * p.queries_per_kv() + warp;
  // the query base for the warp (query_head already contains warp info)
  const std::size_t q_base =
      ((static_cast<std::size_t>(batch)) * p.query_heads + query_head) * p.head_dim;
  // each warp is responsible for one query head
  float4 const *global_q = reinterpret_cast<float4 const *>(in.query + q_base);
  // the actual q this thread is responsible for
  const float4 q = is_valid_lane ? global_q[lane] : make_float4(0, 0, 0, 0);

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float4 accumulator = make_float4(0, 0, 0, 0);

  const int pages = (in.context_lengths[batch] + p.page_size - 1) / p.page_size;
  for (int logical_page = 0; logical_page < pages; ++logical_page) {
    const int physical_page = in.block_tables[batch * p.max_pages_per_sequence + logical_page];
    int valid_tokens = min(p.page_size, in.context_lengths[batch] - logical_page * p.page_size);
    const std::size_t kv_base =
        ((static_cast<std::size_t>(physical_page)) * p.page_size) * p.kv_heads * p.head_dim +
        kv_head * p.head_dim;
    float4 const *global_k4 = reinterpret_cast<float4 const *>(in.key_cache + kv_base);
    float4 const *global_v4 = reinterpret_cast<float4 const *>(in.value_cache + kv_base);

    for (int index = threadIdx.x; index < valid_tokens * chunks; index += blockDim.x) {
      // the step is the block size, so that all threads in the block participate in loading evenly
      int token = index / chunks;
      int chunk = index % chunks;
      shared_k[index] = global_k4[token * p.kv_heads * chunks + chunk];
      shared_v[index] = global_v4[token * p.kv_heads * chunks + chunk];
    }
    __syncthreads();

    for (int token = 0; token < valid_tokens; ++token) {
      // each threads computes a chunk
      float4 k = is_valid_lane ? shared_k[token * chunks + lane] : make_float4(0, 0, 0, 0);
      float4 v = is_valid_lane ? shared_v[token * chunks + lane] : make_float4(0, 0, 0, 0);
      float partial = q.x * k.x + q.y * k.y + q.z * k.z + q.w * k.w;
      float score = warp_reduce_sum(partial);
      score = __shfl_sync(0xffffffff, score, 0);
      score *= in.scale;

      const float new_max = fmaxf(running_max, score);
      const float alpha = running_max == -INFINITY ? 0.0f : __expf(running_max - new_max);
      const float increment = __expf(score - new_max);
      running_sum = running_sum * alpha + increment;
      accumulator.x = accumulator.x * alpha + increment * v.x;
      accumulator.y = accumulator.y * alpha + increment * v.y;
      accumulator.z = accumulator.z * alpha + increment * v.z;
      accumulator.w = accumulator.w * alpha + increment * v.w;
      running_max = new_max;
    }
    __syncthreads();
  }
  accumulator.x /= running_sum;
  accumulator.y /= running_sum;
  accumulator.z /= running_sum;
  accumulator.w /= running_sum;

  const std::size_t out_base = (batch * p.query_heads + query_head) * p.head_dim;
  float4 *out4 = reinterpret_cast<float4 *>(out + out_base);
  if (is_valid_lane) {
    out4[lane] = accumulator;
  }
}

} // namespace
void launch_gqa_reuse_v3(Problem const &p, Inputs const &in, float *out, cudaStream_t stream) {
  // TODO(stage 3): assign a CTA to (batch, kv_head), load a K/V tile once,
  // and process multiple query heads. Measure the reuse/register tradeoff.
  const dim3 grid(p.kv_heads, p.batch);
  const int threads = p.queries_per_kv() * 32; // four warps

  // for each block, allocate one page for k, one page for v
  const int shared_bytes = p.page_size * p.head_dim * 2 * sizeof(float);
  paged_attention_gqa_reuse_kernel<<<grid, threads, shared_bytes, stream>>>(p, in, out);
}
} // namespace pa
