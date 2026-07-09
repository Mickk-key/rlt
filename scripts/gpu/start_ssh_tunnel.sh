#!/usr/bin/env bash
# SSH tunnel: robot localhost:8765 -> fvl08:8765 via fvl05 (10.176.53.120:26570).
# Use when border NAT is not configured yet. Run in a dedicated terminal.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_env.sh
source "$SCRIPT_DIR/../_env.sh"

SSH_HOST="${SSH_HOST:-10.176.53.120}"
SSH_PORT="${SSH_PORT:-26570}"
SSH_USER="${SSH_USER:-yangjiarui}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"
LOCAL_PORT="${LOCAL_PORT:-8765}"
REMOTE_HOST="${REMOTE_HOST:-192.168.110.18}"
REMOTE_PORT="${REMOTE_PORT:-8765}"

PID_FILE="${SMQ_LOG_DIR}/gpu_tunnel_${LOCAL_PORT}.pid"

mkdir -p "$SMQ_LOG_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "[OK] Tunnel already running pid=$(cat "$PID_FILE")"
  exit 0
fi

echo "==> SSH tunnel localhost:${LOCAL_PORT} -> ${REMOTE_HOST}:${REMOTE_PORT}"
echo "    via ${SSH_USER}@${SSH_HOST}:${SSH_PORT}"

ssh -p "$SSH_PORT" -i "$SSH_KEY" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -N -L "${LOCAL_PORT}:${REMOTE_HOST}:${REMOTE_PORT}" \
  "${SSH_USER}@${SSH_HOST}" &
echo $! > "$PID_FILE"
sleep 1

if timeout 2 bash -c "echo >/dev/tcp/127.0.0.1/${LOCAL_PORT}" 2>/dev/null; then
  echo "[OK] Tunnel up. Set: export GPU_SERVER_HOST=127.0.0.1"
else
  echo "[FAIL] Tunnel did not open local port ${LOCAL_PORT}" >&2
  exit 1
fi
