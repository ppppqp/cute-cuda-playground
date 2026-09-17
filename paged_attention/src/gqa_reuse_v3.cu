#include "paged_attention/api.cuh"

namespace pa {
void launch_gqa_reuse_v3(Problem const& p, Inputs const& in, float* out,
                         cudaStream_t stream) {
  // TODO(stage 3): assign a CTA to (batch, kv_head), load a K/V tile once,
  // and process multiple query heads. Measure the reuse/register tradeoff.
  launch_vectorized_v2(p, in, out, stream);
}
}  // namespace pa

