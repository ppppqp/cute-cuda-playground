#pragma once

#include <cuda_runtime.h>

#include "paged_attention/types.cuh"

namespace pa {

void launch_baseline(Problem const &, Inputs const &, float *output, cudaStream_t);
void launch_fused_v1(Problem const &, Inputs const &, float *output, cudaStream_t);
void launch_vectorized_v2(Problem const &, Inputs const &, float *output, cudaStream_t);
void launch_gqa_reuse_v3(Problem const &, Inputs const &, float *output, cudaStream_t);
void launch_split_kv_v4(Problem const &, Inputs const &, float *output,
                        SplitKvWorkspace const &workspace, cudaStream_t);

inline void launch(KernelKind kind, Problem const &problem, Inputs const &inputs, float *output,
                   SplitKvWorkspace const &workspace, cudaStream_t stream = nullptr) {
  switch (kind) {
  case KernelKind::Baseline:
    return launch_baseline(problem, inputs, output, stream);
  case KernelKind::FusedV1:
    return launch_fused_v1(problem, inputs, output, stream);
  case KernelKind::VectorizedV2:
    return launch_vectorized_v2(problem, inputs, output, stream);
  case KernelKind::GqaReuseV3:
    return launch_gqa_reuse_v3(problem, inputs, output, stream);
  case KernelKind::SplitKvV4:
    return launch_split_kv_v4(problem, inputs, output, workspace, stream);
  }
}

} // namespace pa
