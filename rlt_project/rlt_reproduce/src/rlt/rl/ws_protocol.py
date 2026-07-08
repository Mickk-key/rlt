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
    """Compress camera frames to base64 JPEG strings for low-latency transfer."""
    if cv2 is None:
        raise ImportError("opencv-python required for image encoding")
    encoded: dict[str, str] = {}
    for name, frame in images.items():
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
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
