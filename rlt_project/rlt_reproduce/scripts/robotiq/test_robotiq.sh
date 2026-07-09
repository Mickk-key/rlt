#!/usr/bin/env bash
# Test Robotiq 2F gripper (USB-RS485 on PC).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=scripts/_robot_env.sh
source "$ROOT/scripts/_robot_env.sh"

PORT="${ROBOTIQ_PORT:-/dev/ttyUSB0}"
activate_robot_env
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "==> Robotiq test  port=${PORT}"
exec python "$SCRIPT_DIR/run_robotiq_test.py" --port "${PORT}" "$@"
