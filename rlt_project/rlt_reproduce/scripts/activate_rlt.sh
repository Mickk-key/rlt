#!/usr/bin/env bash
# Activate conda rlt env + expose openpi (without breaking rlt editable install).
# Usage: source scripts/activate_rlt.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${ROOT}/third_party/openpi"

if ! command -v conda >/dev/null 2>&1; then
  export PATH="${HOME}/miniconda3/bin:${PATH}"
fi

eval "$(conda shell.bash hook)"
conda activate "${RLT_CONDA_ENV:-rlt}"

export PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${OPENPI_ROOT}/.venv/lib/python3.11/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_ROOT="${ROOT}"
cd "${ROOT}"
