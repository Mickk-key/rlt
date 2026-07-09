#!/usr/bin/env bash
# Release ZMQ client ports (5555/5557) after a crashed teleop or collect session.
set -euo pipefail

echo "==> Checking deoxys Python client (ZMQ port 5555) ..."

# Only one FrankaInterface Python client may bind tcp://*:5555 at a time.
pkill -f "run_deoxys_with_space_mouse.py" 2>/dev/null || true
pkill -f "teleop_spacemouse.py" 2>/dev/null || true
pkill -f "rlt.scripts.collect_plug_insertion" 2>/dev/null || true
pkill -f "collect_plug_insertion.py" 2>/dev/null || true
sleep 0.5

if ss -tlnp 2>/dev/null | grep -q ':5555'; then
  echo "[WARN] Port 5555 still in use — another deoxys client is running:" >&2
  ss -tlnp 2>/dev/null | grep ':5555' || true
  pgrep -af 'FrankaInterface|run_deoxys|teleop_spacemouse|collect_plug' 2>/dev/null || true
  echo "Stop that process (Ctrl+C in its terminal), then retry." >&2
  exit 1
fi

echo "[OK] Port 5555 free — ready for collect/teleop."
