#!/usr/bin/env bash
# List RealSense cameras on USB (no GUI, no deoxys lock).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"

echo "==> RealSense USB enumeration"
if command -v rs-enumerate-devices >/dev/null 2>&1; then
  rs-enumerate-devices 2>/dev/null | awk '
    /^Device info:/ { dev=1; next }
    dev && /Name/ && /RealSense/ { name=$0; sub(/^[^:]*:[[:space:]]*/,"",name); print "  device:", name }
    dev && /Serial Number/ && !/Asic/ { sn=$0; sub(/^[^:]*:[[:space:]]*/,"",sn); print "  serial:", sn; dev=0 }
  '
else
  echo "  rs-enumerate-devices not found"
fi

echo ""
echo "==> pyrealsense2 discovery"
activate_robot_env
python - <<'PY'
import sys
from pathlib import Path

root = Path(__import__("os").environ.get("DEOXYS_ROOT", "/home/host5010/workspaces/wty/deoxys_control/deoxys"))
sys.path.insert(0, str(root))
try:
    from deoxys.camera import RealSenseCameraManager

    serials = RealSenseCameraManager.discover_cameras()
    if not serials:
        print("  [WARN] No RealSense serials found")
    for i, s in enumerate(serials):
        print(f"  [{i}] serial={s}")
    print(f"  total={len(serials)}")
except Exception as exc:
    print(f"  [FAIL] {exc}")
PY

echo ""
echo "==> Stuck camera processes (cause 'Device or resource busy')"
pgrep -af "test_camera|RealSenseCamera|realsense_camera" 2>/dev/null || echo "  none"

echo ""
echo "==> USB RealSense devices"
lsusb 2>/dev/null | grep -i 8086 || echo "  none (check cable / USB3 port)"

echo ""
echo "==> Suggested test command"
activate_robot_env
python - <<'PY'
import sys
from pathlib import Path
root = Path("/home/host5010/workspaces/wty/deoxys_control/deoxys")
sys.path.insert(0, str(root))
from deoxys.camera import RealSenseCameraManager
serials = RealSenseCameraManager.discover_cameras()
if not serials:
    print("  (no cameras — plug into USB3, then re-run)")
else:
    names = ["wrist", "external", "third"]
    mapping = {names[i]: s for i, s in enumerate(serials[:3])}
    import json
    print(f"  CAMERA_MAPPING='{json.dumps(mapping)}' bash scripts/test_camera.sh")
    if len(serials) < 2:
        print("  [WARN] Only 1 camera — plug 2nd into USB3 (not USB2 hub) and re-run")
PY
