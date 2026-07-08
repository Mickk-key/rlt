#!/usr/bin/env bash
# Terminal 1 — 连接机械臂（长线服务，保持终端不关）
# 对应桌面 deoxys.txt Terminal 1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"

activate_robot_env
cd_deoxys_root

INTERFACE_CFG="${1:-$DEOXYS_INTERFACE_CFG}"
echo "==> [Terminal 1] 启动 deoxys 机械臂服务"
echo "    DEOXYS_ROOT=${DEOXYS_ROOT}"
echo "    cfg=${INTERFACE_CFG}"
exec ./auto_scripts/auto_arm.sh "$INTERFACE_CFG"
