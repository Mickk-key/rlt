#!/usr/bin/env bash
# List RealSense cameras on USB (no GUI, no deoxys lock).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"

CONFIG="${1:-$RLT_COLLECT_CONFIG}"

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
echo "==> pyrealsense2 discovery (USB order — may NOT match wrist/external roles)"
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
echo "==> Config mapping (source of truth: ${CONFIG})"
activate_robot_env
python - <<PY
import json, yaml
from pathlib import Path

config = Path("${CONFIG}")
with open(config) as f:
    mapping = yaml.safe_load(f).get("cameras", {}).get("mapping", {})
if mapping:
    for name, sn in mapping.items():
        print(f"  {name}: {sn}")
    print(f"\n  Test: bash scripts/test_camera.sh")
    out = __import__("os").environ.get("SMQ_CAMERA_TEST_DIR", "data/camera_test")
    print(f"  Verify: {out}/wrist.png = wrist mount view")
    print(f"          {out}/external.png = fixed external view")
else:
    print("  [WARN] cameras.mapping empty")
PY

echo ""
echo "==> Stuck camera processes (cause 'Device or resource busy')"
pgrep -af "test_camera|RealSenseCamera|realsense_camera" 2>/dev/null || echo "  none"

echo ""
echo "==> USB RealSense devices"
lsusb 2>/dev/null | grep -i 8086 || echo "  none (check cable / USB3 port)"
