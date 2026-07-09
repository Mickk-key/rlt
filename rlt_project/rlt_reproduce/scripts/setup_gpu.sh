#!/usr/bin/env bash
# Post-install steps after RTX 5090 + NVIDIA driver are ready.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${RLT_CONDA_ENV:-rlt}"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

echo "==> Checking NVIDIA driver..."
if ! nvidia-smi; then
  echo "ERROR: nvidia-smi failed. Install driver first."
  exit 1
fi

echo "==> Reinstalling CUDA PyTorch (adjust cu124/cu126 for your driver)..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "==> openpi uses jax[cuda12]; re-sync if needed:"
echo "    cd ${ROOT}/third_party/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync"

echo "==> Download pi0.5 base checkpoint (~several GB):"
echo "    python ${ROOT}/scripts/download_checkpoint.py --config pi05_droid"

python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
