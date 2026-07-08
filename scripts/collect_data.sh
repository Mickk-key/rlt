#!/usr/bin/env bash
# RLT 任务数据采集（plug insertion critical phase）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"

if ! arm_process_running; then
  echo "ERROR: 机械臂服务未运行。请先:" >&2
  echo "  bash scripts/start_arm.sh" >&2
  echo "  # 或: bash scripts/start_robot.sh" >&2
  exit 1
fi

CONFIG="${1:-$RLT_COLLECT_CONFIG}"

bash "$SCRIPT_DIR/free_cameras.sh"
bash "$SCRIPT_DIR/free_deoxys_client.sh"
bash "$SCRIPT_DIR/check_cameras_config.sh" "$CONFIG"

activate_robot_env
cd "$RLT_ROOT"

echo "==> RLT 数据采集"
echo "    config=${CONFIG}"
echo "    输出: data/episodes/plug_insertion/"
echo ""
echo "【不会自动录制】 r=开始录  s=成功保存并结束  f=失败保存并结束  q=退出"
echo "夹爪: 左键/g 一次夹住保持 | o 手动松开"
echo "SpaceMouse 右键: 机械臂回初始关节位姿(不退出程序)"
echo "无 DISPLAY 时在本终端按 r/s/f/q/g/o"
echo ""

exec python -m rlt.scripts.collect_plug_insertion --config "$CONFIG"
