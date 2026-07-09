"""RealSense camera setup for robot-side rollout (matches collect + actor_loop)."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from rlt.util.deoxys_paths import resolve_deoxys_paths


def build_deoxys_realsense_pair(
    cam_cfg: dict[str, Any],
    *,
    deoxys_root: str | None = None,
    wait_timeout_sec: float = 15.0,
):
    """Open wrist + external RealSense via deoxys ``RealSenseCameraManager``.

    API (deoxys ``realsense_camera.py``):
      ``from deoxys.camera import RealSenseCameraManager``
      ``frames = manager.get_all_latest_frames()`` → ``{name: {rgb, depth, timestamp}}``
    """
    mapping = cam_cfg.get("mapping", {})
    if not mapping:
        return None, {}

    root = Path(deoxys_root) if deoxys_root else Path(resolve_deoxys_paths()[0])
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from deoxys.camera import RealSenseCameraManager

    cam_w = int(cam_cfg.get("width", 640))
    cam_h = int(cam_cfg.get("height", 480))
    cam_fps = int(cam_cfg.get("fps", 30))
    manager = RealSenseCameraManager(
        camera_name_mapping=mapping,
        enable_depth=False,
        fps=cam_fps,
        rgb_resolution=(cam_w, cam_h),
        depth_resolution=(cam_w, cam_h),
    )
    manager.start()
    deadline = time.time() + wait_timeout_sec
    while time.time() < deadline:
        frames = manager.get_all_latest_frames()
        if all(name in frames for name in mapping):
            break
        time.sleep(0.05)
    return manager, mapping


def read_rgb_frames(
    camera_manager,
    camera_mapping: dict[str, str],
) -> dict[str, np.ndarray]:
    """Extract uint8 RGB arrays from ``get_all_latest_frames()``."""
    if camera_manager is None:
        return {}
    frames = camera_manager.get_all_latest_frames()
    images: dict[str, np.ndarray] = {}
    for name in camera_mapping:
        if name not in frames:
            continue
        frame = frames[name]
        rgb = frame["rgb"] if isinstance(frame, dict) else frame
        images[name] = np.asarray(rgb, dtype=np.uint8)
    return images
