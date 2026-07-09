#!/usr/bin/env bash
# Terminal 3 — SpaceMouse 遥操（smq 封装：右键复位不回退出，退出时夹爪松开）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"

if ! arm_process_running; then
  echo "ERROR: 机械臂服务未运行。请先:" >&2
  echo "  bash scripts/start_arm.sh" >&2
  exit 1
fi
if ! gripper_process_running; then
  echo "WARN: 夹爪服务未运行，夹爪控制无效。建议先:" >&2
  echo "  bash scripts/start_gripper.sh" >&2
fi

activate_robot_env

CONTROLLER_TYPE="${1:-$CONTROLLER_TYPE}"
echo "==> SpaceMouse 遥操 (smq)"
echo "    controller=${CONTROLLER_TYPE}"
echo "    左键/g=夹住并保持 | o=松开 | 右键=回初始关节位姿(不退出) | Ctrl+C=退出"
echo "    若末端自己旋转: bash scripts/teleop_test.sh OSC_POSITION"
echo ""

exec python "$SCRIPT_DIR/teleop_spacemouse.py" \
  --interface-cfg "$DEOXYS_INTERFACE_CFG" \
  --controller-type "$CONTROLLER_TYPE" \
  --vendor-id "$SPACEMOUSE_VENDOR_ID" \
  --product-id "$SPACEMOUSE_PRODUCT_ID" \
  --deoxys-root "$DEOXYS_ROOT"
