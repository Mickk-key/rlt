#!/usr/bin/env bash
# Terminal 2 — 连接 Franka 原生夹爪（长线服务，保持终端不关）
# 对应桌面 deoxys.txt Terminal 2
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"

activate_robot_env
cd_deoxys_root

INTERFACE_CFG="${1:-$DEOXYS_INTERFACE_CFG}"
echo "==> [Terminal 2] 启动 deoxys 夹爪服务 (Franka Hand)"
echo "    DEOXYS_ROOT=${DEOXYS_ROOT}"
echo "    cfg=${INTERFACE_CFG}"
echo "    启动后应看到: Gripper homing complete"
exec ./auto_scripts/auto_gripper.sh "$INTERFACE_CFG"
