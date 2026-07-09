#!/usr/bin/env bash
# Test Franka Hand via deoxys (requires arm + gripper-interface running).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=scripts/_robot_env.sh
source "$ROOT/scripts/_robot_env.sh"

activate_robot_env
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if ! arm_process_running; then
  echo "ERROR: deoxys arm not running. Start first:" >&2
  echo "  bash scripts/start_robot.sh" >&2
  exit 1
fi

if ! gripper_process_running; then
  echo "ERROR: gripper-interface not running. Start first:" >&2
  echo "  bash scripts/franka/start_gripper.sh" >&2
  exit 1
fi

exec python "$SCRIPT_DIR/run_franka_gripper_test.py" "$@"
