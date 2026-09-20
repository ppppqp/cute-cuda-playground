#include "paged_attention/api.cuh"

#include <cmath>

namespace pa {
namespace {

// Deliberately inefficient starting point. Each output element redundantly
// recomputes every Q.K dot product. It is simple, deterministic, and correct,
// making it a useful profiling baseline rather than a performance target.
__global__ void paged_attention_baseline_kernel(Problem p, Inputs in, float *__restrict__ out) {
  int const q_head = blockIdx.x;
  int const batch = blockIdx.y;
  int const d_out = threadIdx.x;
  if (d_out >= p.head_dim)
    return;

  int const kv_head = q_head / p.queries_per_kv();
  int const context = in.context_lengths[batch];
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float accumulator = 0.0f;

  for (int token = 0; token < context; ++token) {
    // For each token t, get its cached K(t) and V(t)
    // Assuming for this session, token is continously stored in pages
    // e.g. the ith token is stored in i / page_size th page

    int const logical_page = token / p.page_size;
    int const token_in_page = token % p.page_size;
    // block tables maps token index to the physical page
    // each batch(or "stream") can contain max_pages_per_sequence pages
    /*
    stream 1:   p0,  p1,   p2,   #,  #,  #,  #,  #
    stream 2:   p8,  t9,   t10,  #,  #,  #,  #,  #
    stream 3:   t16, t17,  t18,  #,  #,  #,  #,  #

    # for placeholder (blank page), assuming max_pages_per_sequence=8

    */
    int const physical_page = in.block_tables[batch * p.max_pages_per_sequence + logical_page];

    // this assumes pages are stored continously. therefore the start of the page is
    // physical_page * page_size entries away from the base.
    // Each entry has kv_heads * head_dim fp16
    // Finally add the current kv_head we are computing against, which means another kv_head *
    // head_dim fp16 offset
    std::size_t const kv_base =
        ((static_cast<std::size_t>(physical_page) * p.page_size + token_in_page) * p.kv_heads +
         kv_head) *
        p.head_dim;

    // get the query of the current query token
    std::size_t const q_base =
        (static_cast<std::size_t>(batch) * p.query_heads + q_head) * p.head_dim;

    float score = 0.0f;
    for (int d = 0; d < p.head_dim; ++d) {
      // manually perform dot product of query_head * kv_head
      score = fmaf(in.query[q_base + d], in.key_cache[kv_base + d], score);
    }
    score *= in.scale;

    float const new_max = fmaxf(running_max, score);
    float const old_factor = running_max == -INFINITY ? 0.0f : __expf(running_max - new_max);
    float const new_factor = __expf(score - new_max);
    running_sum = running_sum * old_factor + new_factor;
    accumulator = accumulator * old_factor + new_factor * in.value_cache[kv_base + d_out];
    running_max = new_max;
  }
  // softmax across all tokens, store to the output

  std::size_t const out_index =
      (static_cast<std::size_t>(batch) * p.query_heads + q_head) * p.head_dim + d_out;
  out[out_index] = accumulator / running_sum;
}

} // namespace

void launch_baseline(Problem const &p, Inputs const &in, float *out, cudaStream_t stream) {
  dim3 const grid(p.query_heads, p.batch);
  int const threads = p.head_dim <= 64 ? 64 : 128;
  paged_attention_baseline_kernel<<<grid, threads, 0, stream>>>(p, in, out);
}

} // namespace pa
