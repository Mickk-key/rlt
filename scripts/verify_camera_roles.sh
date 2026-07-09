#!/usr/bin/env bash
# Verify wrist=close-up / external=wide-view via live frame heuristic.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"

CONFIG="${1:-${SFT_COLLECT_CONFIG:-${SMQ_ROOT}/configs/sft_plug_insertion.yaml}}"

bash "$SCRIPT_DIR/free_cameras.sh"
activate_robot_env

export PYTHONPATH="${SMQ_ROOT}/rlt_project/rlt_reproduce/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python "$SCRIPT_DIR/verify_camera_roles.py" --config "$CONFIG" "$@"
