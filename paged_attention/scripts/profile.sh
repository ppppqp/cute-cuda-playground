#!/usr/bin/env bash
set -euo pipefail

build_dir="${BUILD_DIR:-build}"
case_name="${1:-tiny}"
kernel_name="${2:-baseline}"
profile_set="${3:-triage}"
mkdir -p profiles

common=(
  --kernel-name "regex:.*paged_attention.*"
  --launch-skip 10
  --launch-count 1
  --force-overwrite
  -o "profiles/${kernel_name}_${case_name}_${profile_set}"
)

if [[ "$profile_set" == "triage" ]]; then
  ncu --set basic "${common[@]}" \
    "$build_dir/pa_benchmark" --case "$case_name" --kernel "$kernel_name" --iterations 1 --check
elif [[ "$profile_set" == "memory" ]]; then
  ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --section Occupancy --section SchedulerStats --section WarpStateStats \
    "${common[@]}" \
    "$build_dir/pa_benchmark" --case "$case_name" --kernel "$kernel_name" --iterations 1
elif [[ "$profile_set" == "roofline" ]]; then
  ncu --set roofline "${common[@]}" \
    "$build_dir/pa_benchmark" --case "$case_name" --kernel "$kernel_name" --iterations 1
else
  echo "profile set must be triage, memory, or roofline" >&2
  exit 2
fi

