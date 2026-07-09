#!/usr/bin/env bash
# 快速 reset 到采集初始位姿（插头正上方 init cube 内随机点，直达无分阶段）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"

if ! arm_process_running; then
  echo "ERROR: 机械臂服务未运行。请先: bash scripts/start_arm.sh" >&2
  exit 1
fi

CONFIG="${1:-${SFT_COLLECT_CONFIG:-${SMQ_ROOT}/configs/sft_plug_insertion.yaml}}"
FIXED="${FIXED:-0}"
RANDOM="${RANDOM_RESET:-1}"

bash "$SCRIPT_DIR/free_deoxys_client.sh"
activate_robot_env
cd "$RLT_ROOT"

ARGS=(--config "$CONFIG")
if [[ "$FIXED" == "1" ]]; then
  ARGS+=(--fixed)
elif [[ "$RANDOM" == "1" ]]; then
  ARGS+=(--random)
fi

echo "==> 快速 reset 到采集初始位姿 (fast_reset)"
echo "    config=${CONFIG}"
echo ""

exec python -m rlt.scripts.reset_to_init_pose "${ARGS[@]}"
