#include "paged_attention/api.cuh"
#include <cmath>
namespace pa {

namespace {

__device__ float warp_reduce_sum(float value) {
  // each warp has 32 lanes
  for (int offset = 16; offset > 0; offset /= 2) {
    // 0xfffffff means all lanes participate
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void paged_attention_fused_kernel(Problem p, Inputs in, float *out) {
  int const q_head = blockIdx.x;
  int const batch = blockIdx.y;

  int const d = threadIdx.x;
  int const lane = threadIdx.x & 31;
  int const warp = threadIdx.x >> 5;
  const int num_warps = blockDim.x / 32;

  // each thread computes one dimension, and reduce to become dot product
  if (d >= p.head_dim) {
    return;
  }
  int const kv_head = q_head / p.queries_per_kv();

  const std::size_t q_base =
      ((static_cast<std::size_t>(batch)) * p.query_heads + q_head) * p.head_dim;
  float q = in.query[q_base + d];
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float accumulator = 0.0f;

  const int num_pages = (in.context_lengths[batch] + p.page_size - 1) / p.page_size;
  for (int logical_page = 0; logical_page < num_pages; ++logical_page) {
    const int physical_page = in.block_tables[batch * p.max_pages_per_sequence + logical_page];
    int valid_tokens = min(p.page_size, in.context_lengths[batch] - logical_page * p.page_size);
    for (int token = 0; token < valid_tokens; ++token) {
      // load K, V
      const std::size_t kv_base =
          ((static_cast<std::size_t>(physical_page)) * p.page_size + token) * p.kv_heads *
              p.head_dim +
          kv_head * p.head_dim;
      // only load d dimension
      float k_d = in.key_cache[kv_base + d];
      float v_d = in.value_cache[kv_base + d];
      float partial_sum = q * k_d;
      float warp_sum = warp_reduce_sum(partial_sum);
      // assume head_dim = 128, we need 4 warps
      __shared__ float warp_sums[4];
      // check if I am the leader warp
      if (lane == 0) {
        warp_sums[warp] = warp_sum;
      }
      __syncthreads();
      // warp 0 reduces all warps
      float block_sum = 0.0f;
      __shared__ float shared_score;
      if (warp == 0) {
        block_sum = lane < num_warps ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
          shared_score = block_sum * in.scale;
        }
      }
      __syncthreads();
      float const new_max = fmaxf(running_max, shared_score);
      float const old_factor = running_max == -INFINITY ? 0.0f : __expf(running_max - new_max);
      float const increment = __expf(shared_score - new_max);
      // new factor is the incremental
      // prev_sum * old_max / new_max + increment / new_max
      running_sum = running_sum * old_factor + increment;
      // NOTE THAT accumulator is different for each thread
      // because v_d is different
      accumulator = accumulator * old_factor + increment * v_d;
      running_max = new_max;
    }
  }

  std::size_t const out_index = (batch * p.query_heads + q_head) * p.head_dim + d;
  out[out_index] = accumulator / running_sum;
}
} // namespace
void launch_fused_v1(Problem const &p, Inputs const &in, float *out, cudaStream_t stream) {
  // TODO(stage 1): replace this forwarding call with a cooperative CTA kernel.
  // Compute each Q.K once per warp/CTA, use warp reductions, then broadcast the
  // score while lanes update disjoint output dimensions with online softmax.
  dim3 const grid(p.query_heads, p.batch);
  int const threads = p.head_dim <= 64 ? 64 : 128;
  paged_attention_fused_kernel<<<grid, threads, 0, stream>>>(p, in, out);
}
} // namespace pa
