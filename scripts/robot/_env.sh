#!/usr/bin/env bash
# Robot PC env for online RL actor (sources SMQ&JGY root _env.sh).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_env.sh
source "$SCRIPT_DIR/../_env.sh"

export GPU_SERVER_HOST="${GPU_SERVER_HOST:-}"
export GPU_SERVER_PORT="${GPU_SERVER_PORT:-8765}"
export GPU_SERVER_MOCK="${GPU_SERVER_MOCK:-0}"

echo "[robot] SMQ_ROOT=${SMQ_ROOT}"
echo "[robot] DEOXYS_ROOT=${DEOXYS_ROOT}"
echo "[robot] RLT_COLLECT_CONFIG=${RLT_COLLECT_CONFIG}"
echo "[robot] GPU_SERVER_HOST=${GPU_SERVER_HOST:-<unset>}"
