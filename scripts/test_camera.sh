#!/usr/bin/env bash
# Test RealSense cameras. Uses headless mode when no DISPLAY (SSH-safe).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"

bash "$SCRIPT_DIR/free_cameras.sh"

activate_robot_env
ensure_smq_dirs

# Default: auto-build mapping from connected cameras (override with CAMERA_MAPPING env)
if [[ -z "${CAMERA_MAPPING:-}" ]]; then
  CAMERA_MAPPING="$(python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/host5010/workspaces/wty/deoxys_control/deoxys")
from deoxys.camera import RealSenseCameraManager
serials = RealSenseCameraManager.discover_cameras()
names = ["wrist", "external", "third"]
print(json.dumps({names[i]: s for i, s in enumerate(serials[:3])}))
PY
)"
  echo "Auto camera mapping: ${CAMERA_MAPPING}"
fi

if [[ -z "${DISPLAY:-}" ]]; then
  echo "==> No DISPLAY — headless camera test (saves PNG to ${SMQ_CAMERA_TEST_DIR}/)"
  echo "    Tip: bash scripts/detect_cameras.sh  to list actual serials first"
  exec python "$SCRIPT_DIR/test_camera_headless.py" \
    --camera-mapping "$CAMERA_MAPPING" \
    "$@"
fi

cd_deoxys_root
echo "==> RealSense GUI preview (local display detected)"
exec python examples/test_camera.py \
  --camera-mapping "$CAMERA_MAPPING" \
  "$@"
