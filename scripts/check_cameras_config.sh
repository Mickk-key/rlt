#!/usr/bin/env bash
# Verify plug_insertion.yaml camera serials match currently connected RealSense devices.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"

CONFIG="${1:-$RLT_COLLECT_CONFIG}"
activate_robot_env

python - <<PY
import json
import sys
from pathlib import Path

import yaml

config_path = Path("${CONFIG}")
with open(config_path) as f:
    raw = yaml.safe_load(f)

mapping = raw.get("cameras", {}).get("mapping", {})
if not mapping:
    sys.exit(0)

deoxys_root = Path("${DEOXYS_ROOT}")
sys.path.insert(0, str(deoxys_root))
from deoxys.camera import RealSenseCameraManager

connected = set(RealSenseCameraManager.discover_cameras())
missing = {name: sn for name, sn in mapping.items() if sn not in connected}

if missing:
    print("ERROR: config serial(s) not connected:", file=sys.stderr)
    for name, sn in missing.items():
        print(f"  {name}: {sn}", file=sys.stderr)
    print(f"Connected now: {sorted(connected)}", file=sys.stderr)
    print("Run: bash scripts/detect_cameras.sh", file=sys.stderr)
    print("Then update cameras.mapping in plug_insertion.yaml", file=sys.stderr)
    sys.exit(1)

print(f"[OK] All camera serials present: {mapping}")
PY
