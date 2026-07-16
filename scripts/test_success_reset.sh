#!/usr/bin/env bash
# Standalone real-robot test for the post-success reset path (z-lift then归位).
# Wraps scripts/test_success_reset.py with the same env as the actor loop
# (activates franka_mani, sets PYTHONPATH / DEOXYS paths) so it never runs in
# the wrong conda env.
#
#   bash scripts/test_success_reset.sh --dry-run
#   CONFIRM=1 bash scripts/test_success_reset.sh --config configs/plug_insertion.yaml
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"

DRY_RUN=0
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && DRY_RUN=1
done

CONFIRM="${CONFIRM:-0}"

# Real motion needs arm service + a free ZMQ client (matches run_actor_loop.sh).
if [[ "$DRY_RUN" != "1" && "$CONFIRM" == "1" ]]; then
  if ! arm_process_running; then
    echo "ERROR: 机械臂服务未运行。请先: bash scripts/start_arm.sh" >&2
    exit 1
  fi
  if ! gripper_process_running; then
    echo "WARN: 夹爪服务未运行。建议: bash scripts/start_gripper.sh" >&2
  fi
  bash "$SCRIPT_DIR/free_deoxys_client.sh"
fi

activate_robot_env
# Match run_actor_loop.sh: relative config paths (gripper.config etc.) resolve under RLT_ROOT.
cd "$RLT_ROOT"

echo "==> post-success reset test (z-lift then归位)"
echo "    dry_run=${DRY_RUN} CONFIRM=${CONFIRM}"
echo ""

exec python "$SCRIPT_DIR/test_success_reset.py" "$@"
