#!/usr/bin/env bash
# 关节复位
# 对应桌面 deoxys.txt Terminal 7 reset
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"

activate_robot_env
cd_deoxys_root

echo "==> 机械臂关节复位 (deoxys.reset_joints)"
exec deoxys.reset_joints "$@"
