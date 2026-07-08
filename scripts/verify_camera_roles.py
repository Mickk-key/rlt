#!/usr/bin/env python3
"""Verify yaml camera mapping: wrist=close-up, external=fixed wide view."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

SMQ_ROOT = Path(os.environ.get("SMQ_ROOT", Path(__file__).resolve().parents[1]))
RLT_SRC = SMQ_ROOT / "rlt_project" / "rlt_reproduce" / "src"
if str(RLT_SRC) not in sys.path:
    sys.path.insert(0, str(RLT_SRC))

DEOXYS_ROOT = Path(os.environ.get("DEOXYS_ROOT", SMQ_ROOT / "third_party/deoxys"))
if str(DEOXYS_ROOT) not in sys.path:
    sys.path.insert(0, str(DEOXYS_ROOT))

from deoxys.camera import RealSenseCameraManager  # noqa: E402
from rlt.util.realsense_rgb import read_rgb_png, verify_wrist_external_roles, write_rgb_png  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify wrist/external camera role mapping")
    p.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("SFT_COLLECT_CONFIG", SMQ_ROOT / "configs/sft_plug_insertion.yaml")),
    )
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument(
        "--ref-dir",
        type=Path,
        default=Path(os.environ.get("SMQ_CAMERA_TEST_DIR", SMQ_ROOT / "data/camera_test")),
        help="Reference wrist.png / external.png from bash scripts/test_camera.sh",
    )
    return p.parse_args()


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
    with open(args.config) as f:
        raw = yaml.safe_load(f)
    cam = raw.get("cameras", {})
    mapping = cam.get("mapping", {})
    if not mapping:
        print("[FAIL] cameras.mapping empty in config")
        return 1

    w = int(cam.get("width", 640))
    h = int(cam.get("height", 480))
    fps = int(cam.get("fps", 30))

    manager = RealSenseCameraManager(
        camera_name_mapping=mapping,
        enable_depth=False,
        fps=fps,
        rgb_resolution=(w, h),
        depth_resolution=(w, h),
    )
    try:
        manager.start()
        names = list(mapping.keys())
        frames_raw = wait_frames(manager, names, args.timeout)
        rgb_frames = {k: frames_raw[k]["rgb"] for k in names}

        references: dict = {}
        for key in ("wrist", "external"):
            ref_path = args.ref_dir / f"{key}.png"
            if ref_path.is_file():
                references[key] = read_rgb_png(ref_path)
        if not references:
            print(f"[WARN] No reference PNGs in {args.ref_dir} — run: bash scripts/test_camera.sh")

        ok, msg = verify_wrist_external_roles(rgb_frames, references=references or None)
        print(f"[{'OK' if ok else 'FAIL'}] {msg}")
        print(f"  mapping={json.dumps(mapping)}")

        if args.save_dir:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            for name, rgb in rgb_frames.items():
                write_rgb_png(args.save_dir / f"{name}.png", rgb)
            print(f"  debug PNGs -> {args.save_dir}")

        if not ok:
            print(
                "  Fix: swap serials under cameras.mapping in sft_plug_insertion.yaml, then re-run "
                "bash scripts/test_camera.sh",
                file=sys.stderr,
            )
            return 1
        return 0
    finally:
        manager.stop()


if __name__ == "__main__":
    raise SystemExit(main())
