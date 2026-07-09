"""Build openpi observation dicts from websocket robot payloads."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

OBS_IMAGE_SIZE = (224, 224)
PROPRIO_GRIPPER_INDEX = 7


def resize_rgb(img: np.ndarray, size: tuple[int, int] = OBS_IMAGE_SIZE) -> np.ndarray:
    arr = np.asarray(img)
    if arr.shape[:2] != size:
        if cv2 is None:
            raise ImportError("opencv-python required for image resize")
        arr = cv2.resize(arr, size, interpolation=cv2.INTER_AREA)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def proprio_to_openpi_state(proprio: np.ndarray, mapping: dict[str, Any]) -> dict[str, np.ndarray]:
    """Map 8-dim EE pose proprio into openpi DroidInputs slots."""
    joint_idx = mapping.get("joint_slice", list(range(7)))
    gripper_idx = int(mapping.get("gripper_index", PROPRIO_GRIPPER_INDEX))
    joint = np.asarray(proprio)[joint_idx].astype(np.float32)
    gripper = np.asarray([proprio[gripper_idx]], dtype=np.float32)
    return {
        "observation/joint_position": joint,
        "observation/gripper_position": gripper,
    }


def ws_obs_to_openpi(
    proprio: np.ndarray,
    images: dict[str, np.ndarray] | None,
    language: str,
    *,
    proprio_mapping: dict[str, Any],
    image_size: tuple[int, int] = OBS_IMAGE_SIZE,
) -> dict[str, Any]:
    """Convert actor websocket obs → openpi infer dict."""
    state = proprio_to_openpi_state(proprio, proprio_mapping)
    obs: dict[str, Any] = {**state, "prompt": language or ""}
    if images:
        external = images.get("external")
        wrist = images.get("wrist")
        if external is not None:
            obs["observation/exterior_image_1_left"] = resize_rgb(external, image_size)
        if wrist is not None:
            obs["observation/wrist_image_left"] = resize_rgb(wrist, image_size)
    return obs


def ws_obs_to_libero(
    proprio: np.ndarray,
    images: dict[str, np.ndarray] | None,
    language: str,
    *,
    proprio_mapping: dict[str, Any] | None = None,  # noqa: ARG001 (libero state is raw proprio)
    image_size: tuple[int, int] = OBS_IMAGE_SIZE,
) -> dict[str, Any]:
    """Convert actor websocket obs → openpi LiberoInputs dict.

    Must match the SFT input format exactly (see data.plug_insertion.frame_to_libero_obs):
    8-dim state [ee_pos(3), ee_quat(4), gripper_width(1)] + base/wrist images, keyed for
    openpi LiberoInputs (observation/image, observation/wrist_image, observation/state).
    """
    obs: dict[str, Any] = {
        "observation/state": np.asarray(proprio, dtype=np.float32),
        "prompt": language or "",
    }
    if images:
        external = images.get("external")
        wrist = images.get("wrist")
        if external is not None:
            obs["observation/image"] = resize_rgb(external, image_size)
        if wrist is not None:
            obs["observation/wrist_image"] = resize_rgb(wrist, image_size)
    return obs


# "droid" -> base pi05_droid format; "libero" -> finetuned pi05_plug_insertion format.
_OBS_BUILDERS = {
    "droid": ws_obs_to_openpi,
    "libero": ws_obs_to_libero,
}


def build_observation_for_format(
    input_format: str,
    proprio: np.ndarray,
    images: dict[str, np.ndarray] | None,
    language: str,
    *,
    proprio_mapping: dict[str, Any],
    image_size: tuple[int, int] = OBS_IMAGE_SIZE,
) -> dict[str, Any]:
    builder = _OBS_BUILDERS.get(str(input_format).lower(), ws_obs_to_openpi)
    return builder(
        proprio, images, language, proprio_mapping=proprio_mapping, image_size=image_size
    )


# Alias used in docs / worker-side code.
build_openpi_observation = ws_obs_to_openpi
