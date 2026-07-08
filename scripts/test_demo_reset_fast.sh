#!/usr/bin/env bash
# Time demo_fast reset (skip joint home when near target). Run before long online RL sessions.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"

CONFIG="${1:-$RLT_COLLECT_CONFIG}"
TRIALS="${TRIALS:-5}"
PAUSE_SEC="${PAUSE_SEC:-1.0}"
CONFIRM="${CONFIRM:-0}"

if [[ "$CONFIRM" != "1" ]]; then
  echo "ERROR: 需要 CONFIRM=1。示例:" >&2
  echo "  CONFIRM=1 TRIALS=5 bash scripts/test_demo_reset_fast.sh" >&2
  exit 1
fi

if ! arm_process_running; then
  echo "ERROR: bash scripts/start_arm.sh  first" >&2
  exit 1
fi

bash "$SCRIPT_DIR/free_deoxys_client.sh"
activate_robot_env
cd "$RLT_ROOT"

echo "==> demo_fast reset 计时（6cm 内跳过 motion；小位移直走 xyz）"
echo "    config=${CONFIG}  trials=${TRIALS}"
echo ""

exec python -m rlt.scripts.test_demo_reset \
  --config "$CONFIG" \
  --trials "$TRIALS" \
  --pause-sec "$PAUSE_SEC" \
  --mode env \
  --reset-mode demo_fast
