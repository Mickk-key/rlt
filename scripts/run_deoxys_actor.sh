#!/usr/bin/env bash
# Robot-side online RL actor: sources robot/_env.sh then run_actor_loop.sh.
set -euo pipefail
SMQ_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=robot/_env.sh
source "$SMQ_SCRIPTS_DIR/robot/_env.sh"
if [[ -f "$SMQ_SCRIPTS_DIR/robot/deoxys_actor.env" ]]; then
  # shellcheck source=robot/deoxys_actor.env
  source "$SMQ_SCRIPTS_DIR/robot/deoxys_actor.env"
fi
exec bash "$SMQ_SCRIPTS_DIR/run_actor_loop.sh" "$@"
