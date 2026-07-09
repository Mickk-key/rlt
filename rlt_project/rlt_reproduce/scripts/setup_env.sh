#!/usr/bin/env bash
# One-shot environment setup for RLT reproduction (CPU-first; GPU optional).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_DIR="${ROOT}/third_party/openpi"
CONDA_ENV="${RLT_CONDA_ENV:-rlt}"

echo "==> RLT setup (root: ${ROOT})"

# 1. Conda env with Python 3.11
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  echo "==> Creating conda env: ${CONDA_ENV} (python=3.11)"
  conda create -y -n "${CONDA_ENV}" python=3.11 pip
fi

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# 2. Install this project (CPU torch if no GPU)
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "==> CUDA detected — installing default torch"
  pip install -e "${ROOT}[dev]"
else
  echo "==> No CUDA — installing CPU-only PyTorch"
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  pip install -e "${ROOT}[dev]"
fi

# 3. openpi (VLA backbone) via uv
if [[ -d "${OPENPI_DIR}" ]]; then
  echo "==> Syncing openpi dependencies (this may take several minutes)..."
  cd "${OPENPI_DIR}"
  git submodule update --init --recursive
  if command -v uv >/dev/null 2>&1 || [[ -x "${CONDA_PREFIX}/../bin/uv" ]] || [[ -x "/home/host5010/miniconda3/bin/uv" ]]; then
    UV_BIN="$(command -v uv 2>/dev/null || echo "/home/host5010/miniconda3/bin/uv")"
    export UV_HTTP_TIMEOUT=600
    GIT_LFS_SKIP_SMUDGE=1 "${UV_BIN}" venv --python 3.11 .venv 2>/dev/null || true
    GIT_LFS_SKIP_SMUDGE=1 "${UV_BIN}" sync || {
      echo "WARN: uv sync failed (often network). Retry:"
      echo "  cd ${OPENPI_DIR} && UV_HTTP_TIMEOUT=600 GIT_LFS_SKIP_SMUDGE=1 ${UV_BIN} sync"
    }
    # Expose openpi to conda env
    pip install -e "${OPENPI_DIR}"
    pip install -e "${OPENPI_DIR}/packages/openpi-client"
  else
    echo "WARN: uv not found; install openpi manually: https://github.com/Physical-Intelligence/openpi"
  fi
  cd "${ROOT}"
else
  echo "WARN: openpi not found at ${OPENPI_DIR}"
  echo "      Clone with: git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git third_party/openpi"
fi

echo ""
echo "==> Running verification..."
python -m rlt.scripts.verify_env

echo ""
echo "Setup complete. Activate with: conda activate ${CONDA_ENV}"
echo "Smoke test: python -m rlt.scripts.smoke_test"
