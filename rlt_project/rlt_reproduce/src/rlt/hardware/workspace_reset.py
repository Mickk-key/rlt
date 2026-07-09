"""Random workspace reset — same entry as ``reset_to_init.sh`` / SFT collection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rlt.hardware.deoxys.collection_reset import collection_reset_settings, reset_to_init_cube
from rlt.hardware.deoxys.fast_reset import InitCubeConfig

WorkspaceRandomizationConfig = InitCubeConfig


@dataclass
class WorkspaceResetResult:
    target_xyz: np.ndarray
    offset_xy: np.ndarray
    offset_xyz: np.ndarray
    episode_id: str
    reset_info: dict


def reset_random_workspace(
    robot_interface,
    *,
    gripper,
    joint_controller_cfg,
    osc_position_cfg,
    ws_cfg: InitCubeConfig,
    demo_reset_yaml: dict | None = None,
    sft_cfg: dict | None = None,
    raw: dict | None = None,
    pos_tol_m: float = 0.015,
    logger=None,
) -> WorkspaceResetResult:
    del demo_reset_yaml, ws_cfg, sft_cfg, pos_tol_m
    if raw is None:
        raise ValueError("reset_random_workspace requires full config dict as raw=")
    result = reset_to_init_cube(
        robot_interface,
        gripper=gripper,
        osc_position_cfg=osc_position_cfg,
        joint_controller_cfg=joint_controller_cfg,
        raw=raw,
        randomize=True,
        logger=logger,
    )
    _, fast_cfg = collection_reset_settings(raw)
    offset = result.offset_xyz
    return WorkspaceResetResult(
        target_xyz=result.target_xyz,
        offset_xy=offset[:2],
        offset_xyz=offset,
        episode_id="sft_random",
        reset_info={
            "success": result.pos_err_m <= fast_cfg.pos_tol_m * 2.0,
            "steps": result.steps,
            "joint_home_used": result.joint_home_used,
            "motion_skipped": result.motion_skipped,
            "pos_err_m": result.pos_err_m,
            "collection_reset": True,
        },
    )
