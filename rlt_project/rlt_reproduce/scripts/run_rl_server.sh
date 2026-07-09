#!/usr/bin/env bash
# GPU-side RL server (run on the training host with CUDA).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${RLT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="${1:-${RLT_ROOT}/configs/franka/fr3_franka.yaml}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"
DEVICE="${DEVICE:-cuda}"

cd "$RLT_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"

echo "==> RLT GPU Server"
echo "    config=${CONFIG}"
echo "    ws://${HOST}:${PORT}"
echo "    device=${DEVICE}"
echo ""

exec python -m rlt.scripts.rl_server --config "$CONFIG" --host "$HOST" --port "$PORT" --device "$DEVICE"
