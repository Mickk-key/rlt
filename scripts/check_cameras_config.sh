#!/usr/bin/env bash
# Verify camera serials + optional role/RGB check on data/camera_test/*.png
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"

CONFIG="${1:-${SFT_COLLECT_CONFIG:-${RLT_COLLECT_CONFIG:-${SMQ_ROOT}/configs/sft_plug_insertion.yaml}}}"
activate_robot_env

python - <<PY
import sys
from pathlib import Path
import yaml

config_path = Path("${CONFIG}")
with open(config_path) as f:
    mapping = yaml.safe_load(f).get("cameras", {}).get("mapping", {})
if not mapping:
    sys.exit(0)

deoxys_root = Path("${DEOXYS_ROOT}")
sys.path.insert(0, str(deoxys_root))
from deoxys.camera import RealSenseCameraManager

connected = set(RealSenseCameraManager.discover_cameras())
missing = {n: s for n, s in mapping.items() if s not in connected}
if missing:
    print("ERROR: config serial(s) not connected:", file=sys.stderr)
    for n, s in missing.items():
        print(f"  {n}: {s}", file=sys.stderr)
    print(f"Connected: {sorted(connected)}", file=sys.stderr)
    sys.exit(1)
print(f"[OK] serials present: {mapping}")

test_dir = Path("${SMQ_CAMERA_TEST_DIR}")
w, e = test_dir / "wrist.png", test_dir / "external.png"
if not (w.is_file() and e.is_file()):
    print(f"[SKIP] role check — run: bash scripts/test_camera.sh")
    sys.exit(0)

import cv2
import numpy as np

def center_dark_frac(path):
    g = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    c = g[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    return float((c < 45).mean())

def rb_means(path):
    bgr = cv2.imread(str(path))
    return float(bgr[:, :, 2].mean()), float(bgr[:, :, 0].mean())

dw, de = center_dark_frac(w), center_dark_frac(e)
if dw <= de:
    print("ERROR: wrist.png 更像外置视角 — 请对调 cameras.mapping 后重跑 test_camera.sh", file=sys.stderr)
    print(f"  dark_frac wrist={dw:.3f} external={de:.3f}", file=sys.stderr)
    sys.exit(1)
rw, bw = rb_means(w)
if bw > rw + 25:
    print(f"WARN: wrist.png 可能 R/B 反了 (R={rw:.0f} B={bw:.0f})")
else:
    print(f"[OK] wrist 近景特征 dark={dw:.3f}>{de:.3f}, R={rw:.0f}>=B={bw:.0f}")
print("[OK] camera_test 角色与 RGB 正常")
PY
