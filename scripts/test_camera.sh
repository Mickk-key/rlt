#!/usr/bin/env bash
# Test RealSense cameras. Uses headless mode when no DISPLAY (SSH-safe).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"

bash "$SCRIPT_DIR/free_cameras.sh"

activate_robot_env
ensure_smq_dirs

# Default: read cameras.mapping from RLT_COLLECT_CONFIG (NOT USB enumeration order).
if [[ -z "${CAMERA_MAPPING:-}" ]]; then
  CAMERA_MAPPING="$(python - <<PY
import json, os, yaml
from pathlib import Path
config = Path(os.environ["RLT_COLLECT_CONFIG"])
with open(config) as f:
    mapping = yaml.safe_load(f).get("cameras", {}).get("mapping", {})
if not mapping:
    raise SystemExit("cameras.mapping empty in " + str(config))
print(json.dumps(mapping))
PY
)"
  echo "Camera mapping from ${RLT_COLLECT_CONFIG}:"
  echo "  ${CAMERA_MAPPING}"
  echo "  (USB order may differ — verify wrist.png vs external.png visually)"
fi

if [[ -z "${DISPLAY:-}" ]]; then
  echo "==> No DISPLAY — headless camera test (saves PNG to ${SMQ_CAMERA_TEST_DIR}/)"
  echo "    Tip: bash scripts/detect_cameras.sh  to list USB serials"
  exec python "$SCRIPT_DIR/test_camera_headless.py" \
    --camera-mapping "$CAMERA_MAPPING" \
    "$@"
fi

cd_deoxys_root
echo "==> RealSense GUI preview (local display detected)"
exec python examples/test_camera.py \
  --camera-mapping "$CAMERA_MAPPING" \
  "$@"
