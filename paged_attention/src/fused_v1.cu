#include "paged_attention/api.cuh"

namespace pa {
void launch_fused_v1(Problem const& p, Inputs const& in, float* out,
                     cudaStream_t stream) {
  // TODO(stage 1): replace this forwarding call with a cooperative CTA kernel.
  // Compute each Q.K once per warp/CTA, use warp reductions, then broadcast the
  // score while lanes update disjoint output dimensions with online softmax.
  launch_baseline(p, in, out, stream);
}
}  // namespace pa

