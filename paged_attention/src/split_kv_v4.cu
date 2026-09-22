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

__global__ void paged_attention_split_kernel(Problem p, Inputs in, float *global_partial_max,
                                             float *global_partial_sum,
                                             float *global_partial_output, int num_splits) {
  int const q_head = blockIdx.x;
  int const batch = blockIdx.y;
  int const split = blockIdx.z;

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
  const int num_pages_per_split = (num_pages + split - 1) / num_splits;

  for (int logical_page = num_pages_per_split * split;
       logical_page < num_pages_per_split * (split + 1); ++logical_page) {
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
  std::size_t const out_base =
      (batch * p.query_heads * num_splits + p.query_heads * split + q_head) * p.head_dim;
  float *split_partial_max =
      global_partial_max + batch * p.query_heads * num_splits + p.query_heads * split + q_head;
  float *split_partial_sum =
      global_partial_sum + batch * p.query_heads * num_splits + p.query_heads * split + q_head;
  float4 *split_partial_output = reinterpret_cast<float4 *>(global_partial_output + out_base);
  if (lane == 0) {
    // partial max and partial sum are per-cta
    split_partial_max[0] = running_max;
    split_partial_sum[0] = running_sum;
  }
  if (is_valid_lane) {
    split_partial_output[lane] = accumulator;
  }
}
__global__ void paged_attention_merge_kernel(Problem p, Inputs in, float *global_partial_max,
                                             float *global_partial_sum,
                                             float *global_partial_output, int num_splits,
                                             float *out) {
  const int q_head = blockIdx.x;
  const int batch = blockIdx.y;
  constexpr int vec_width = 4;
  const int chunks = p.head_dim / vec_width;
  const int lane = threadIdx.x & 31;
  const bool is_valid_lane = lane < chunks;

  float combined_max = -INFINITY;
  float combined_sum = 0.0f;
  float4 combined_output = make_float4(0, 0, 0, 0);

  for (int split = 0; split < num_splits; ++split) {
    float split_max = 0.0f;
    float split_sum = 0.0f;
    int state_index = batch * p.query_heads * num_splits + p.query_heads * split + q_head;
    if (lane == 0) {
      split_max = global_partial_max[state_index];
      split_sum = global_partial_sum[state_index];
    }
    split_max = __shfl_sync(0xffffffff, split_max, 0);
    split_sum = __shfl_sync(0xffffffff, split_sum, 0);
    float4 *global_partial_output4 = reinterpret_cast<float4 *>(global_partial_output);
    float4 split_output = is_valid_lane ? global_partial_output4[state_index * chunks + lane]
                                        : make_float4(0, 0, 0, 0);

    if (split_sum != 0.0f) {
      // NOTE: if split_sum is 0.0f, it means the split has no element
      // merging it will lead to -INF / -INF error
      const float new_max = fmaxf(combined_max, split_max);
      const float alpha = combined_max == -INFINITY ? 0.0f : __expf(combined_max - new_max);
      // have to use alpha-beta here, because we have no idea of the token level score now
      // so we can not do increment = __expf(score - new_max);
      // have to directly adjust the already calculated split_output
      const float beta = __expf(split_max - new_max);
      combined_sum = alpha * combined_sum + beta * split_sum;
      combined_output.x = combined_output.x * alpha + beta * split_output.x;
      combined_output.y = combined_output.y * alpha + beta * split_output.y;
      combined_output.z = combined_output.z * alpha + beta * split_output.z;
      combined_output.w = combined_output.w * alpha + beta * split_output.w;
      combined_max = new_max;
    }
  }
  combined_output.x /= combined_sum;
  combined_output.y /= combined_sum;
  combined_output.z /= combined_sum;
  combined_output.w /= combined_sum;
  if (is_valid_lane) {
    float4 *out4 = reinterpret_cast<float4 *>(out + (batch * p.query_heads + q_head) * p.head_dim);
    out4[lane] = combined_output;
  }
}

void launch_split_kv_partial(Problem const &p, Inputs const &in, SplitKvWorkspace const &workspace,
                             cudaStream_t stream) {
  const dim3 grid(p.query_heads, p.batch, workspace.num_splits);
  const int threads = 32;

  paged_attention_split_kernel<<<grid, threads, 0, stream>>>(
      p, in, workspace.partial_max, workspace.partial_sum, workspace.partial_output,
      workspace.num_splits);
}
void launch_split_kv_merge(Problem const &p, Inputs const &in, SplitKvWorkspace const &workspace,
                           cudaStream_t stream, float *out) {
  const dim3 grid(p.query_heads, p.batch);
  const int threads = 32;
  paged_attention_merge_kernel<<<grid, threads, 0, stream>>>(
      p, in, workspace.partial_max, workspace.partial_sum, workspace.partial_output,
      workspace.num_splits, out);
}
} // namespace

void launch_split_kv_v4(Problem const &p, Inputs const &in, float *out,
                        SplitKvWorkspace const &workspace, cudaStream_t stream) {
  // TODO(stage 4): split long contexts across CTAs. Store each split's max,
  // sum, and partial output, then combine them with log-sum-exp correction.
  // Add workspace ownership to the public API when implementing this stage.

  launch_split_kv_partial(p, in, workspace, stream);
  // no need to sync between them if they are launched into the same CUDA stream
  launch_split_kv_merge(p, in, workspace, stream, out);
}
} // namespace pa
