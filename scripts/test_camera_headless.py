#!/usr/bin/env python3
"""Headless RealSense check — no GUI required (SSH-safe).

Grabs one frame per camera and saves PNGs under SMQ&JGY/data/camera_test/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SMQ_ROOT = Path(os.environ.get("SMQ_ROOT", Path(__file__).resolve().parents[1]))
DEFAULT_OUT = Path(
    os.environ.get("SMQ_CAMERA_TEST_DIR", SMQ_ROOT / "data" / "camera_test")
)

DEOXYS_ROOT = Path(os.environ.get("DEOXYS_ROOT", SMQ_ROOT / "third_party/deoxys"))
if not (DEOXYS_ROOT / "deoxys" / "camera").is_dir() and not (DEOXYS_ROOT / "deoxys").is_dir():
    legacy = Path("/home/host5010/workspaces/wty/deoxys_control/deoxys")
    if legacy.is_dir():
        DEOXYS_ROOT = legacy
if str(DEOXYS_ROOT) not in sys.path:
    sys.path.insert(0, str(DEOXYS_ROOT))

from deoxys.camera import RealSenseCameraManager  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless RealSense camera test")
    parser.add_argument(
        "--camera-mapping",
        type=str,
        default="",
        help='JSON dict name->serial, e.g. \'{"wrist":"244222070454"}\'. Empty = auto-discover.',
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
    )
    parser.add_argument("--timeout", type=float, default=15.0)
    return parser.parse_args()


def wait_frames(manager: RealSenseCameraManager, names: list[str], timeout: float) -> dict:
    deadline = time.time() + timeout
    latest: dict = {}
    while time.time() < deadline:
        latest.update(manager.get_all_latest_frames())
        if all(n in latest for n in names):
            return latest
        time.sleep(0.05)
    missing = [n for n in names if n not in latest]
    raise RuntimeError(f"Timed out waiting for cameras: {missing}")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mapping = None
    if args.camera_mapping.strip():
        mapping = json.loads(args.camera_mapping)
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("camera-mapping must be a non-empty JSON object")

    manager = RealSenseCameraManager(
        camera_name_mapping=mapping,
        enable_depth=False,
        fps=args.fps,
        rgb_resolution=(args.width, args.height),
        depth_resolution=(args.width, args.height),
    )

    try:
        manager.start()
        names = list(manager.get_camera_info().keys())
        if not names:
            print("[FAIL] No RealSense cameras opened.")
            return 1

        print(f"[OK] Opened cameras: {names}")
        for name, info in manager.get_camera_info().items():
            serial = mapping[name] if mapping else "auto"
            print(f"  - {name}: serial={serial}  intrinsics={info}")

        frames = wait_frames(manager, names, args.timeout)
        ok = True
        for name in names:
            rgb = np.asarray(frames[name]["rgb"], dtype=np.uint8)
            bgr = rgb[:, :, ::-1] if rgb.ndim == 3 and rgb.shape[2] == 3 else rgb
            out = args.out_dir / f"{name}.png"
            cv2.imwrite(str(out), bgr)
            mean = float(np.mean(rgb))
            print(f"[OK] {name}: shape={rgb.shape} mean={mean:.1f} -> {out}")
            if mean < 1.0:
                print(f"[WARN] {name} frame looks black — check lens cap / exposure")
                ok = False

        print(f"\nSaved previews to: {args.out_dir}")
        return 0 if ok else 2
    finally:
        manager.stop()


if __name__ == "__main__":
    raise SystemExit(main())
