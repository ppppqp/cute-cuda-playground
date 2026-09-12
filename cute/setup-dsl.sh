#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 -m venv "${project_dir}/.venv"
"${project_dir}/.venv/bin/python" -m pip install \
  -r "${project_dir}/third_party/cutlass/python/CuTeDSL/requirements.txt"

echo "CuTe DSL is installed. Run: source ${project_dir}/dsl-env.sh"
