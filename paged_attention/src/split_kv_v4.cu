#include "paged_attention/api.cuh"

namespace pa {
void launch_split_kv_v4(Problem const& p, Inputs const& in, float* out,
                        cudaStream_t stream) {
  // TODO(stage 4): split long contexts across CTAs. Store each split's max,
  // sum, and partial output, then combine them with log-sum-exp correction.
  // Add workspace ownership to the public API when implementing this stage.
  launch_gqa_reuse_v3(p, in, out, stream);
}
}  // namespace pa

