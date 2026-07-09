#!/usr/bin/env bash
# Demo-driven reset stability test (single robot PC, no GPU learner).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"

CONFIG="${1:-$RLT_COLLECT_CONFIG}"
TRIALS="${TRIALS:-3}"
PAUSE_SEC="${PAUSE_SEC:-2.0}"
MODE="${MODE:-env}"   # env | direct
DRY_RUN="${DRY_RUN:-0}"
CONFIRM="${CONFIRM:-0}"

if [[ "$DRY_RUN" != "1" && "$CONFIRM" != "1" ]]; then
  echo "ERROR: 实机 demo reset 测试需要显式确认。" >&2
  echo "  请先确认工作空间无障碍、急停可用，然后:" >&2
  echo "  CONFIRM=1 bash scripts/test_demo_reset.sh" >&2
  echo "" >&2
  echo "  仅验证数据加载（不连机械臂）:" >&2
  echo "  DRY_RUN=1 bash scripts/test_demo_reset.sh" >&2
  exit 1
fi

if [[ "$DRY_RUN" != "1" ]]; then
  if ! arm_process_running; then
    echo "ERROR: 机械臂服务未运行。请先:" >&2
    echo "  bash scripts/start_arm.sh" >&2
    exit 1
  fi
  if ! gripper_process_running; then
    echo "WARN: 夹爪服务未运行，gripper reset 可能不准。建议:" >&2
    echo "  bash scripts/start_gripper.sh" >&2
  fi
  bash "$SCRIPT_DIR/free_deoxys_client.sh"
fi

activate_robot_env
cd "$RLT_ROOT"

echo "==> Demo reset 稳定性测试（安全模式）"
echo "    config=${CONFIG}"
echo "    trials=${TRIALS}  pause=${PAUSE_SEC}s  mode=${MODE}"
echo ""
echo "安全流程: 关节 home → position_only 到 demo xyz（可选功能，默认关闭）"
echo "安全提示:"
echo "  - 测试前确认工作空间无障碍"
echo "  - 勿同时运行 teleop_test / collect_data"
echo "  - Ctrl+C 可随时中断"
echo ""

ARGS=(--config "$CONFIG" --trials "$TRIALS" --pause-sec "$PAUSE_SEC" --mode "$MODE")
if [[ "$DRY_RUN" == "1" ]]; then
  ARGS+=(--dry-run)
fi

exec python -m rlt.scripts.test_demo_reset "${ARGS[@]}"
