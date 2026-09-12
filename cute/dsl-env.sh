#!/usr/bin/env bash

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="${project_dir}/.venv"

if [[ ! -f "${venv_dir}/bin/activate" ]]; then
  echo "CuTe DSL environment is missing; run ${project_dir}/setup-dsl.sh" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${venv_dir}/bin/activate"
export CUTE_DSL_CACHE_DIR="${project_dir}/.cache/cute_dsl"
export PYTHONUNBUFFERED=1
mkdir -p "${CUTE_DSL_CACHE_DIR}"

unset project_dir venv_dir
