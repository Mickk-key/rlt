#!/usr/bin/env bash
# Ping GPU RL websocket server from robot PC.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"

CONFIG="${1:-$RLT_COLLECT_CONFIG}"
HOST="${GPU_SERVER_HOST:-${2:-}}"

activate_robot_env
cd "$RLT_ROOT"

ARGS=(--config "$CONFIG")
if [[ -n "$HOST" ]]; then
  ARGS+=(--host "$HOST")
fi
if [[ "${MOCK:-0}" == "1" ]]; then
  ARGS+=(--mock)
fi

exec python -m rlt.scripts.ping_gpu_server "${ARGS[@]}"
