#!/usr/bin/env bash
# 记录当前 EE 位姿为 init 立方体底面中心（插座正上方）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"

if ! arm_process_running; then
  echo "ERROR: bash scripts/start_arm.sh first" >&2
  exit 1
fi

CONFIG="${1:-${SFT_COLLECT_CONFIG:-${SMQ_ROOT}/configs/sft_plug_insertion.yaml}}"
WRITE="${WRITE:-0}"

bash "$SCRIPT_DIR/free_deoxys_client.sh"
activate_robot_env
cd "$RLT_ROOT"

ARGS=(--config "$CONFIG")
if [[ "$WRITE" == "1" ]]; then
  ARGS+=(--write)
fi

echo "==> 记录 init 立方体底面中心 (插座正上方 EE 位姿)"
echo "    config=${CONFIG}  WRITE=${WRITE}"
echo ""
echo "先遥操到插座正上方，停止 teleop 后运行。"
echo "确认写 yaml: WRITE=1 bash scripts/record_bottom_center.sh"
echo ""

exec python -m rlt.scripts.record_bottom_center "${ARGS[@]}"
