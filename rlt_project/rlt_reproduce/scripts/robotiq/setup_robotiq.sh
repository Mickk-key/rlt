#!/usr/bin/env bash
# Install Robotiq (Modbus) Python deps for franka_mani / rlt envs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=scripts/_robot_env.sh
source "$ROOT/scripts/_robot_env.sh"

activate_robot_env
pip install pymodbus pyserial
echo "==> Done. Run: bash scripts/robotiq/test_robotiq.sh"
