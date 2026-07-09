#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_env.sh
source "$SCRIPT_DIR/../_env.sh"
PID_FILE="${SMQ_LOG_DIR}/gpu_tunnel_8765.pid"
if [[ -f "$PID_FILE" ]]; then
  kill "$(cat "$PID_FILE")" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "[OK] Stopped tunnel"
else
  echo "No tunnel pid file"
fi
