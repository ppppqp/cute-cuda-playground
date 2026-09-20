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

__global__ void paged_attention_vectorized_kernel(Problem p, Inputs in, float *out) {
  int const q_head = blockIdx.x;
  int const batch = blockIdx.y;

  constexpr int vec_width = 4;
  int const chunks = p.head_dim / vec_width;
  int const lane = threadIdx.x & 31;

  // if head_dim = 64, only 16 lanes will be used
  const bool is_valid_lane = lane < chunks;

  int const kv_head = q_head / p.queries_per_kv();

  const std::size_t q_base =
      ((static_cast<std::size_t>(batch)) * p.query_heads + q_head) * p.head_dim;
  // reintepret Q as float4 aligned
  auto const *query4 = reinterpret_cast<float4 const *>(in.query + q_base);
  // load Q aligned
  float4 q = is_valid_lane ? query4[lane] : make_float4(0, 0, 0, 0);
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float4 accumulator = make_float4(0, 0, 0, 0);

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

      float4 const *key4 = reinterpret_cast<float4 const *>(in.key_cache + kv_base);
      float4 const *value4 = reinterpret_cast<float4 const *>(in.value_cache + kv_base);

      // load d aligned
      float4 k_d = is_valid_lane ? key4[lane] : make_float4(0, 0, 0, 0);
      float4 v_d = is_valid_lane ? value4[lane] : make_float4(0, 0, 0, 0);
      float partial_sum = q.x * k_d.x + q.y * k_d.y + q.z * k_d.z + q.w * k_d.w;
      float shared_score = warp_reduce_sum(partial_sum);
      // need to broad cast to all lanes
      shared_score = __shfl_sync(0xffffffff, shared_score, 0);
      shared_score *= in.scale;
      // assume head_dim = 128, we only need 1 warp
      float const new_max = fmaxf(running_max, shared_score);
      float const old_factor = running_max == -INFINITY ? 0.0f : __expf(running_max - new_max);
      float const increment = __expf(shared_score - new_max);
      // new factor is the incremental
      // prev_sum * old_max / new_max + increment / new_max
      running_sum = running_sum * old_factor + increment;
      accumulator.x = accumulator.x * old_factor + increment * v_d.x;
      accumulator.y = accumulator.y * old_factor + increment * v_d.y;
      accumulator.z = accumulator.z * old_factor + increment * v_d.z;
      accumulator.w = accumulator.w * old_factor + increment * v_d.w;
      running_max = new_max;
    }
  }
  float inverse_sum = 1.0f / running_sum;
  accumulator.x *= inverse_sum;
  accumulator.y *= inverse_sum;
  accumulator.z *= inverse_sum;
  accumulator.w *= inverse_sum;

  std::size_t const out_base = (batch * p.query_heads + q_head) * p.head_dim;
  float4 *out4 = reinterpret_cast<float4 *>(out + out_base);
  if (is_valid_lane) {
    out4[lane] = accumulator;
  }
}

} // namespace

void launch_vectorized_v2(Problem const &p, Inputs const &in, float *out, cudaStream_t stream) {
  // TODO(stage 2): start from fused_v1. Add aligned 16-byte Q/K/V accesses,
  // compare cache layouts, and preserve a scalar tail path if needed.
  dim3 const grid(p.query_heads, p.batch);

  // only need one warp
  int const threads = 32;
  paged_attention_vectorized_kernel<<<grid, threads, 0, stream>>>(p, in, out);
}
} // namespace pa
