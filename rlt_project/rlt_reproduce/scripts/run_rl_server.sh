#!/usr/bin/env bash
# GPU-side RL server (run on the training host with CUDA).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="${RLT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="${1:-${RLT_ROOT}/configs/franka/fr3_franka.yaml}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"
DEVICE="${DEVICE:-cuda}"

cd "$RLT_ROOT"
echo "==> RLT GPU Server"
echo "    config=${CONFIG}"
echo "    ws://${HOST}:${PORT}"
echo "    device=${DEVICE}"
echo ""

exec python -m rlt.scripts.rl_server --config "$CONFIG" --host "$HOST" --port "$PORT" --device "$DEVICE"
