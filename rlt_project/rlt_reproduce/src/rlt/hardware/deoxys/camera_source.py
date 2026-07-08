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
    *,
    cache: dict[str, np.ndarray] | None = None,
    allow_stale: bool = False,
) -> dict[str, np.ndarray]:
    """Extract uint8 RGB arrays from ``get_all_latest_frames()``."""
    if camera_manager is None:
        return {}
    frames = camera_manager.get_all_latest_frames()
    images: dict[str, np.ndarray] = {}
    for name in camera_mapping:
        if name not in frames:
            if allow_stale and cache is not None and name in cache:
                images[name] = cache[name]
            continue
        frame = frames[name]
        rgb = frame["rgb"] if isinstance(frame, dict) else frame
        arr = np.asarray(rgb, dtype=np.uint8)
        images[name] = arr
        if cache is not None:
            cache[name] = arr
    return images


def wait_for_rgb_frames(
    camera_manager,
    camera_mapping: dict[str, str],
    *,
    timeout_sec: float = 5.0,
    cache: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Wait until every camera has at least one frame in *cache* (prefer new shm frames).

    deoxys ``get_all_latest_frames()`` only returns *new* frames since the last read.
    Requiring all cameras to publish a new frame on the same poll often fails (e.g. wrist
    lags external). We therefore accumulate into *cache* and return once every mapped
    camera has been seen at least once — stale wrist + fresh external is OK for RL.
    """
    if cache is None:
        cache = {}
    names = list(camera_mapping)
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        read_rgb_frames(camera_manager, camera_mapping, cache=cache)
        if all(name in cache for name in names):
            return {name: cache[name] for name in names}
        time.sleep(0.02)
    read_rgb_frames(camera_manager, camera_mapping, cache=cache, allow_stale=True)
    if all(name in cache for name in names):
        return {name: cache[name] for name in names}
    missing = sorted(set(names) - set(cache))
    raise TimeoutError(f"Cameras missing frames after {timeout_sec}s: {missing}")
