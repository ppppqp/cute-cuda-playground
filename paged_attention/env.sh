#!/usr/bin/env bash

# Source this file from bash. It reuses the CUDA selector in ~/.bashrc and
# keeps all build products inside this project.
if ! declare -F activate-cuda >/dev/null; then
  # ~/.bashrc returns early in non-interactive shells, so extract the function
  # by starting an interactive bash when necessary.
  eval "$(bash -ic 'declare -f activate-cuda' 2>/dev/null)"
fi

if ! declare -F activate-cuda >/dev/null; then
  echo "activate-cuda was not found in ~/.bashrc" >&2
  return 1 2>/dev/null || exit 1
fi

activate-cuda
export CUTE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CUDA 12.8 supports GCC through version 13; this machine's unversioned GCC is
# newer. CMake and nvcc both honor these variables.
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export CUDAHOSTCXX=/usr/bin/g++-13