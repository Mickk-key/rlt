#!/usr/bin/env bash
# GPU host (fvl08 / 10.176.53.120): JPEG WebSocket RL server for robot actor_loop.
# Keep this terminal open while dual-machine rollout is running.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "${ROOT}/scripts/activate_rlt.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"

CONFIG="${1:-${ROOT}/configs/plug_insertion_gpu.yaml}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"
DEVICE="${DEVICE:-cuda}"

echo "==> smq&jgy JPEG RL server (GPU)"
echo "    root=${ROOT}"
echo "    config=${CONFIG}"
echo "    ws://${HOST}:${PORT}  (robot connects to 10.176.53.120:${PORT})"
echo "    device=${DEVICE}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo ""

exec bash "${ROOT}/scripts/run_rl_server.sh" "${CONFIG}"
