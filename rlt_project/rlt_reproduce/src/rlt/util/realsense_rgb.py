"""RealSense RGB image helpers (deoxys streams rgb8 — never treat as BGR)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def as_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """Ensure HWC RGB uint8 from RealSense / recorder pipeline."""
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected RGB HWC uint8, got shape={arr.shape} dtype={arr.dtype}")
    return arr


def resize_rgb(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize RGB frame; size is (width, height) for cv2.resize."""
    return cv2.resize(as_rgb_uint8(img), size)


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(as_rgb_uint8(img), cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_BGR2RGB)


def write_rgb_png(path: str | Path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), rgb_to_bgr(rgb))


def read_rgb_png(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return bgr_to_rgb(bgr)


def center_laplacian_variance(rgb: np.ndarray) -> float:
    """Sharpness proxy on center crop."""
    gray = cv2.cvtColor(as_rgb_uint8(rgb), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    crop = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    return float(cv2.Laplacian(crop, cv2.CV_64F).var())


def _resize_for_compare(a: np.ndarray, b: np.ndarray, size: tuple[int, int] = (320, 240)) -> tuple[np.ndarray, np.ndarray]:
    aa = cv2.resize(as_rgb_uint8(a), size).astype(np.float32)
    bb = cv2.resize(as_rgb_uint8(b), size).astype(np.float32)
    return aa, bb


def mse_rgb(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = _resize_for_compare(a, b)
    return float(np.mean((aa - bb) ** 2))


def match_roles_against_references(
    frames: dict[str, np.ndarray],
    references: dict[str, np.ndarray],
    *,
    wrist_key: str = "wrist",
    external_key: str = "external",
) -> tuple[bool, str]:
    """Each live stream should be closest (lowest MSE) to its namesake reference PNG."""
    missing = [k for k in (wrist_key, external_key) if k not in frames]
    if missing:
        return False, f"missing live frames: {missing}"
    missing_ref = [k for k in (wrist_key, external_key) if k not in references]
    if missing_ref:
        return False, f"missing reference PNGs: {missing_ref}"

    details = []
    ok = True
    for cam_name in (wrist_key, external_key):
        live = frames[cam_name]
        scores = {ref_name: mse_rgb(live, references[ref_name]) for ref_name in references}
        best_ref = min(scores, key=scores.get)
        details.append(f"{cam_name}->best={best_ref} mse={scores}")
        if best_ref != cam_name:
            ok = False
    msg = "; ".join(details)
    return ok, msg


def verify_wrist_external_roles(
    frames: dict[str, np.ndarray],
    *,
    wrist_key: str = "wrist",
    external_key: str = "external",
    references: dict[str, np.ndarray] | None = None,
    min_ratio: float = 1.05,
) -> tuple[bool, str]:
    """Verify mapping using reference PNGs when available, else spatial heuristics."""
    if references:
        return match_roles_against_references(
            frames, references, wrist_key=wrist_key, external_key=external_key
        )

    if wrist_key not in frames or external_key not in frames:
        return False, f"missing frames (have {sorted(frames.keys())})"

    # Fallback: external fixed cam sees robot arm on the left; wrist cam targets gripper lower-center.
    def _region_lap(rgb: np.ndarray, y0: float, y1: float, x0: float, x1: float) -> float:
        gray = cv2.cvtColor(as_rgb_uint8(rgb), cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        crop = gray[int(h * y0) : int(h * y1), int(w * x0) : int(w * x1)]
        return float(cv2.Laplacian(crop, cv2.CV_64F).var())

    w_left = _region_lap(frames[wrist_key], 0.0, 1.0, 0.0, 0.35)
    e_left = _region_lap(frames[external_key], 0.0, 1.0, 0.0, 0.35)
    w_bot = _region_lap(frames[wrist_key], 0.45, 1.0, 0.25, 0.75)
    e_bot = _region_lap(frames[external_key], 0.45, 1.0, 0.25, 0.75)

    external_ok = e_left >= w_left * min_ratio
    wrist_ok = w_bot >= e_bot * min_ratio
    msg = (
        f"heuristic left(wrist={w_left:.1f}, ext={e_left:.1f}) "
        f"bottom(wrist={w_bot:.1f}, ext={e_bot:.1f})"
    )
    if external_ok and wrist_ok:
        return True, msg
    return False, f"roles likely swapped: {msg}"
