#!/usr/bin/env bash
# Release RealSense after a crashed test_camera.py (fixes 'Device or resource busy').
set -euo pipefail

echo "==> Killing stuck camera python processes ..."
pkill -f "examples/test_camera.py" 2>/dev/null || true
pkill -f "test_camera_headless.py" 2>/dev/null || true
pkill -f "realsense_camera.py" 2>/dev/null || true
sleep 1

if pgrep -f "test_camera|realsense_camera" >/dev/null 2>&1; then
  echo "[WARN] Some camera processes still running:"
  pgrep -af "test_camera|realsense_camera" || true
else
  echo "[OK] No stuck camera processes."
fi
