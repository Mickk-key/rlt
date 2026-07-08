"""Random workspace reset — thin wrapper over fast_reset.init cube logic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rlt.hardware.deoxys.fast_reset import (
    FastResetConfig,
    InitCubeConfig,
    reset_to_collection_init,
)

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
    pos_tol_m: float = 0.015,
    logger=None,
) -> WorkspaceResetResult:
    del demo_reset_yaml
    sc = sft_cfg or {}
    fast_cfg = FastResetConfig(
        pos_tol_m=pos_tol_m,
        control_hz=float(sc.get("fps", 50.0)),
        approach_xy_first=bool(sc.get("approach_xy_first", True)),
        joint_home_if_delta_above_m=float(sc.get("joint_home_if_delta_above_m", 0.35)),
    )
    result = reset_to_collection_init(
        robot_interface,
        gripper=gripper,
        cube_cfg=ws_cfg,
        osc_position_cfg=osc_position_cfg,
        joint_controller_cfg=joint_controller_cfg,
        reset_cfg=fast_cfg,
        randomize=True,
        logger=logger,
    )
    return WorkspaceResetResult(
        target_xyz=result.target_xyz,
        offset_xy=result.offset_xyz[:2],
        offset_xyz=result.offset_xyz,
        episode_id="sft_random",
        reset_info={
            "success": result.pos_err_m <= fast_cfg.pos_tol_m * 2,
            "steps": result.steps,
            "joint_home_used": result.joint_home_used,
            "motion_skipped": result.motion_skipped,
            "pos_err_m": result.pos_err_m,
            "fast_reset": True,
        },
    )
