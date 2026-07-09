"""Shared websocket payload helpers for robot PC ↔ GPU RL server."""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def resize_rgb_frames(
    images: dict[str, np.ndarray],
    size: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Resize uint8 RGB frames to (W, H) — matches collect_plug_insertion / openpi 224."""
    if cv2 is None:
        raise ImportError("opencv-python required for image resize")
    w, h = int(size[0]), int(size[1])
    out: dict[str, np.ndarray] = {}
    for name, frame in images.items():
        arr = np.asarray(frame, dtype=np.uint8)
        if arr.shape[1] == w and arr.shape[0] == h:
            out[name] = arr
        else:
            out[name] = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
    return out


def encode_images_jpeg(images: dict[str, np.ndarray], *, quality: int = 80) -> dict[str, str]:
    """Compress uint8 RGB camera frames to base64 JPEG strings for websocket transfer."""
    if cv2 is None:
        raise ImportError("opencv-python required for image encoding")
    encoded: dict[str, str] = {}
    for name, frame in images.items():
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        # RealSense / read_rgb_frames are RGB; cv2.imencode expects BGR.
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError(f"JPEG encode failed for camera {name}")
        encoded[name] = base64.b64encode(buf.tobytes()).decode("ascii")
    return encoded


def decode_images_jpeg(images_b64: dict[str, str]) -> dict[str, np.ndarray]:
    if cv2 is None:
        raise ImportError("opencv-python required for image decoding")
    out: dict[str, np.ndarray] = {}
    for name, blob in images_b64.items():
        raw = base64.b64decode(blob.encode("ascii"))
        arr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise RuntimeError(f"JPEG decode failed for camera {name}")
        # Return RGB uint8 (matches read_rgb_frames / RealSense convention).
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        out[name] = arr
    return out


def pack_observation(
    proprio: np.ndarray,
    *,
    images: dict[str, np.ndarray] | None = None,
    language: str = "",
    jpeg_quality: int = 80,
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Build infer payload (proprio always; images optional as JPEG base64)."""
    payload: dict[str, Any] = {
        "proprio": proprio.astype(np.float32),
        "language": language,
    }
    if images:
        frames = images
        if image_size is not None:
            frames = resize_rgb_frames(images, image_size)
        payload["images_jpeg"] = encode_images_jpeg(frames, quality=jpeg_quality)
    return payload


REQUIRED_CAMERA_KEYS = ("external", "wrist")


def ensure_observation_images_jpeg(
    observation: dict[str, Any],
    *,
    image_size: tuple[int, int] | None = None,
    jpeg_quality: int = 80,
    required_keys: tuple[str, ...] = REQUIRED_CAMERA_KEYS,
) -> dict[str, Any]:
    """Attach ``images_jpeg`` from raw RGB ``observation['images']`` for GPU infer."""
    out = dict(observation)
    if "images_jpeg" in out and out["images_jpeg"]:
        missing = [k for k in required_keys if k not in out["images_jpeg"]]
        if missing:
            raise ValueError(
                f"observation images_jpeg missing keys {missing}; "
                f"got {sorted(out['images_jpeg'])}"
            )
        return out

    images = out.get("images")
    if not images:
        return out

    frames = {k: np.asarray(v, dtype=np.uint8) for k, v in images.items()}
    if image_size is not None:
        frames = resize_rgb_frames(frames, image_size)
    out["images_jpeg"] = encode_images_jpeg(frames, quality=jpeg_quality)

    missing = [k for k in required_keys if k not in out["images_jpeg"]]
    if missing:
        raise ValueError(
            f"observation images missing required keys {missing}; got {sorted(images)}"
        )
    return out
