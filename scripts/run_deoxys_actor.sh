#!/usr/bin/env bash
# Robot-side online RL actor: sources robot/_env.sh then run_actor_loop.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=robot/_env.sh
source "$SCRIPT_DIR/robot/_env.sh"
if [[ -f "$SCRIPT_DIR/robot/deoxys_actor.env" ]]; then
  # shellcheck source=robot/deoxys_actor.env
  source "$SCRIPT_DIR/robot/deoxys_actor.env"
fi
exec bash "$SCRIPT_DIR/run_actor_loop.sh" "$@"
