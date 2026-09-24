#!/usr/bin/env bash
set -euo pipefail

build_dir="${BUILD_DIR:-build}"
case_name="${1:-tiny}"
kernel_name="${2:-baseline}"
profile_set="${3:-triage}"
num_splits="${4:-8}"
mkdir -p profiles

# Most variants launch one matching kernel per benchmark iteration. Split-KV
# launches a partial kernel followed by a merge kernel, so its ten warm-up
# iterations contain twenty matching launches. Profile both kernels from the
# first timed iteration.
kernel_filter="regex:.*paged_attention.*"
launch_skip=10
launch_count=1
benchmark_args=(
  "$build_dir/pa_benchmark"
  --case "$case_name"
  --kernel "$kernel_name"
  --iterations 1
)

case "$kernel_name" in
  baseline)
    kernel_filter="regex:.*paged_attention_baseline_kernel.*"
    ;;
  fused_v1)
    kernel_filter="regex:.*paged_attention_fused_kernel.*"
    ;;
  vectorized_v2)
    kernel_filter="regex:.*paged_attention_vectorized_kernel.*"
    ;;
  gqa_reuse_v3)
    kernel_filter="regex:.*paged_attention_gqa_reuse_kernel.*"
    ;;
  split_kv_v4)
    if [[ ! "$num_splits" =~ ^[1-9][0-9]*$ ]]; then
      echo "num_splits must be a positive integer" >&2
      exit 2
    fi
    kernel_filter="regex:.*paged_attention_(split|merge)_kernel.*"
    launch_skip=20
    launch_count=2
    benchmark_args+=(--num-splits "$num_splits")
    ;;
  *)
    echo "unknown kernel: $kernel_name" >&2
    exit 2
    ;;
esac

common=(
  --kernel-name "$kernel_filter"
  --launch-skip "$launch_skip"
  --launch-count "$launch_count"
  --force-overwrite
  -o "profiles/${kernel_name}_${case_name}_${profile_set}"
)

if [[ "$profile_set" == "triage" ]]; then
  ncu --set basic "${common[@]}" \
    "${benchmark_args[@]}" --check
elif [[ "$profile_set" == "memory" ]]; then
  ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --section Occupancy --section SchedulerStats --section WarpStateStats \
    "${common[@]}" \
    "${benchmark_args[@]}"
elif [[ "$profile_set" == "roofline" ]]; then
  ncu --set roofline "${common[@]}" \
    "${benchmark_args[@]}"
else
  echo "profile set must be triage, memory, or roofline" >&2
  exit 2
fi
