#!/usr/bin/env bash
# 机械臂 + RealSense 同步采集
# 对应桌面 deoxys.txt 底部 collect_low_dim_states_with_realsense_sync.py
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"

if ! arm_process_running; then
  echo "ERROR: 机械臂服务未运行。请先: bash scripts/start_arm.sh" >&2
  exit 1
fi

activate_robot_env
cd_deoxys_root

echo "==> 机械臂 + RealSense 同步采集"
exec python examples/collect_low_dim_states_with_realsense_sync.py "$@"
