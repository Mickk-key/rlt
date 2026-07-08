#!/usr/bin/env bash
# Terminal 4 — 采集低维机械臂数据（不含相机）
# 对应桌面 deoxys.txt Terminal 4
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

echo "==> [Terminal 4] 低维数据采集 (SpaceMouse, 无相机)"
exec python examples/collect_low_dim_states_with_spacemouse.py "$@"
