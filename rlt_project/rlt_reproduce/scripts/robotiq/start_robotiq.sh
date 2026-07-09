#!/usr/bin/env bash
# Activate Robotiq 2F gripper over USB-RS485 (Modbus).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=scripts/_robot_env.sh
source "$ROOT/scripts/_robot_env.sh"

activate_robot_env
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

port="$(resolve_robotiq_port)"
if [[ ! -e "$port" ]]; then
  echo "ERROR: Robotiq serial port not found: $port" >&2
  exit 1
fi

echo "==> Robotiq activation test on ${port}"
export ROBOTIQ_PORT="$port"
exec python "$SCRIPT_DIR/run_robotiq_test.py" --config "$ROBOTIQ_CONFIG" --port "$port"
