#include "paged_attention/api.cuh"

namespace pa {
void launch_vectorized_v2(Problem const& p, Inputs const& in, float* out,
                          cudaStream_t stream) {
  // TODO(stage 2): start from fused_v1. Add aligned 16-byte Q/K/V accesses,
  // compare cache layouts, and preserve a scalar tail path if needed.
  launch_fused_v1(p, in, out, stream);
}
}  // namespace pa

