#!/usr/bin/env bash
# Local loopback: rl_server (CPU) + actor_loop mock env over websocket.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"

CONFIG="${1:-$RLT_COLLECT_CONFIG}"
PORT="${GPU_SERVER_PORT:-8765}"
EPISODES="${EPISODES:-1}"
MAX_STEPS="${MAX_STEPS:-15}"

activate_robot_env
cd "$RLT_ROOT"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "==> Starting local rl_server on 127.0.0.1:${PORT} (CPU) ..."
python -m rlt.scripts.rl_server --config "$CONFIG" --host 127.0.0.1 --port "$PORT" --device cpu &
SERVER_PID=$!
sleep 2

echo "==> Ping GPU server ..."
GPU_SERVER_HOST=127.0.0.1 GPU_SERVER_PORT="$PORT" GPU_SERVER_MOCK=0 \
  python -m rlt.scripts.ping_gpu_server --config "$CONFIG" --host 127.0.0.1

echo "==> Actor loop (mock env, real websocket, ${EPISODES} ep, max ${MAX_STEPS} steps) ..."
GPU_SERVER_HOST=127.0.0.1 GPU_SERVER_PORT="$PORT" GPU_SERVER_MOCK=0 \
  MAX_STEPS="$MAX_STEPS" EPISODES="$EPISODES" \
  python -m rlt.scripts.actor_loop --mock --config "$CONFIG" --gpu-host 127.0.0.1 \
    --episodes "$EPISODES" --max-steps "$MAX_STEPS"

echo "[OK] Loopback test passed"
