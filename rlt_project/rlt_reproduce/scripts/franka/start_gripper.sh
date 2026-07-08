#!/usr/bin/env bash
# Start deoxys Franka Hand gripper service (long-running; keep terminal open or use start_robot.sh).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=scripts/_robot_env.sh
source "$ROOT/scripts/_robot_env.sh"

activate_robot_env
cd_deoxys_root

INTERFACE_CFG="${1:-$DEOXYS_INTERFACE_CFG}"

if gripper_process_running; then
  echo "==> deoxys gripper-interface already running (skipping start)"
  exit 0
fi

echo "==> Starting deoxys gripper-interface in background ..."
echo "    cfg=${INTERFACE_CFG}"
echo "    log=${RLT_GRIPPER_LOG}"
nohup ./auto_scripts/auto_gripper.sh "$INTERFACE_CFG" >"$RLT_GRIPPER_LOG" 2>&1 &
echo $! >"$RLT_GRIPPER_PID_FILE"
echo "    pid=$(cat "$RLT_GRIPPER_PID_FILE")"
sleep 3
