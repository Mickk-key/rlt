"""Fast EE reset for collection / demo init (no multi-segment hover pipeline).

Inspired by deoxys ``experimental.motion_utils.position_only_gripper_move_to``:
single OSC_POSITION loop straight to target xyz.

Reference pose = **bottom center** of the init randomization cube (directly above plug).
Random sample: xy ± half_range on bottom face, z in [0, z_range_m] above bottom.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from rlt.hardware.deoxys.demo_reset import ResetPose, _wait_for_state
from rlt.teleop.spacemouse_control import DEFAULT_RESET_JOINTS


def _resolve_log(logger=None):
    """Accept stdlib Logger, rich Console, or None."""
    if logger is None:
        return print
    if callable(getattr(logger, "info", None)):
        return logger.info
    if callable(getattr(logger, "print", None)):
        return logger.print
    return print


@dataclass
class InitCubeConfig:
    """Random init volume for plug-insertion collection."""

    enabled: bool = True
    bottom_center_xyz: tuple[float, float, float] = (0.677, -0.016, 0.189)
    reference_quat: tuple[float, float, float, float] = (-0.012, 0.997, -0.059, 0.045)
    gripper_width: float = 0.0
    xy_half_range_m: float = 0.05
    z_range_m: float = 0.0
    min_z_m: float | None = None  # defaults to bottom_center z; never command below
    seed: int | None = None

    @property
    def floor_z(self) -> float:
        if self.min_z_m is not None:
            return float(self.min_z_m)
        return float(self.bottom_center_xyz[2])

    @classmethod
    def from_yaml_dict(cls, ws: dict) -> InitCubeConfig:
        bottom = ws.get("bottom_center_xyz") or ws.get("reference_ee_pos") or [0.677, -0.016, 0.189]
        half = float(ws.get("xy_half_range_m", ws.get("xy_range_m", 0.10) / 2.0))
        min_z = ws.get("min_z_m")
        return cls(
            enabled=bool(ws.get("enabled", True)),
            bottom_center_xyz=tuple(bottom),
            reference_quat=tuple(ws.get("reference_quat", [-0.012, 0.997, -0.059, 0.045])),
            gripper_width=float(ws.get("gripper_width", 0.0)),
            xy_half_range_m=half,
            z_range_m=float(ws.get("z_range_m", 0.0)),
            min_z_m=float(min_z) if min_z is not None else None,
            seed=ws.get("seed"),
        )


@dataclass
class FastResetConfig:
    control_hz: float = 50.0
    pos_tol_m: float = 0.015
    xy_tol_m: float = 0.012
    max_steps: int = 200
    min_steps: int = 25
    gripper_closed: float = 1.0
    joint_home_if_delta_above_m: float = 0.35
    joint_home_timeout: float = 10.0
    home_joints: list[float] = field(default_factory=lambda: list(DEFAULT_RESET_JOINTS))
    skip_if_within_m: float = 0.008
    approach_xy_first: bool = True  # xy at current z, then z down — stay above plug


@dataclass
class FastResetResult:
    target_xyz: np.ndarray
    offset_xyz: np.ndarray
    steps: int
    joint_home_used: bool
    motion_skipped: bool
    pos_err_m: float


def sample_init_pose_in_cube(cfg: InitCubeConfig, rng: np.random.Generator) -> ResetPose:
    bc = np.asarray(cfg.bottom_center_xyz, dtype=np.float64)
    floor_z = cfg.floor_z
    if cfg.enabled:
        ox = float(rng.uniform(-cfg.xy_half_range_m, cfg.xy_half_range_m))
        oy = float(rng.uniform(-cfg.xy_half_range_m, cfg.xy_half_range_m))
        oz = float(rng.uniform(0.0, max(0.0, cfg.z_range_m)))
        target = bc + np.array([ox, oy, oz], dtype=np.float64)
        target[2] = max(target[2], floor_z)
        offset = target - bc
    else:
        target = bc.copy()
        target[2] = max(target[2], floor_z)
        offset = np.zeros(3, dtype=np.float64)

    return ResetPose(
        ee_pose=target.astype(np.float32),
        quaternion=np.asarray(cfg.reference_quat, dtype=np.float32),
        gripper_width=float(cfg.gripper_width),
        episode_id="init_cube_random" if cfg.enabled else "init_cube_fixed",
        success=None,
    )


def _current_pos(robot_interface) -> np.ndarray:
    while True:
        if robot_interface._state_buffer:
            from rlt.hardware.deoxys_arm import o_t_ee_to_pose

            st = robot_interface._state_buffer[-1]
            pos, _ = o_t_ee_to_pose(st.O_T_EE)
            return np.asarray(pos, dtype=np.float64).reshape(3)
        time.sleep(0.02)


def _maybe_joint_home(
    robot_interface,
    target_xyz: np.ndarray,
    *,
    joint_controller_cfg,
    reset_cfg: FastResetConfig,
) -> bool:
    pos = _current_pos(robot_interface)
    delta = float(np.linalg.norm(target_xyz - pos))
    if delta <= reset_cfg.joint_home_if_delta_above_m:
        return False
    from deoxys.experimental.motion_utils import reset_joints_to

    reset_joints_to(
        robot_interface,
        list(reset_cfg.home_joints),
        controller_cfg=joint_controller_cfg,
        timeout=reset_cfg.joint_home_timeout,
        gripper_open=False,
    )
    return True


def fast_move_to_xyz(
    robot_interface,
    target_xyz: np.ndarray,
    *,
    osc_position_cfg,
    reset_cfg: FastResetConfig | None = None,
    gripper_cmd: float | None = None,
    lock_axes: str | None = None,
    floor_z: float | None = None,
    tol_m: float | None = None,
) -> tuple[int, float]:
    """Direct OSC_POSITION move. lock_axes: None | 'xy' | 'z'."""
    reset_cfg = reset_cfg or FastResetConfig()
    gripper_cmd = reset_cfg.gripper_closed if gripper_cmd is None else gripper_cmd
    target_xyz = np.asarray(target_xyz, dtype=np.float64).reshape(3).copy()
    if floor_z is not None:
        target_xyz[2] = max(target_xyz[2], floor_z)
    tol = reset_cfg.pos_tol_m if tol_m is None else tol_m

    _wait_for_state(robot_interface)
    pos0 = _current_pos(robot_interface)
    dist = float(np.linalg.norm(target_xyz - pos0))
    max_steps = int(
        np.clip(
            reset_cfg.min_steps + dist / 0.002,
            reset_cfg.min_steps,
            reset_cfg.max_steps,
        )
    )
    dt = 1.0 / reset_cfg.control_hz
    steps = 0

    while steps < max_steps:
        current_pos = _current_pos(robot_interface)
        err = target_xyz - current_pos
        if lock_axes == "xy":
            err[2] = 0.0
        elif lock_axes == "z":
            err[0] = err[1] = 0.0
        if lock_axes == "xy":
            pos_err = float(np.linalg.norm(err[:2]))
        elif lock_axes == "z":
            pos_err = abs(float(err[2]))
        else:
            pos_err = float(np.linalg.norm(err))
        if pos_err <= tol:
            return steps, pos_err

        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(err * 10.0, -1.0, 1.0).astype(np.float32)
        action[6] = float(gripper_cmd)
        robot_interface.control(
            controller_type="OSC_POSITION",
            action=action,
            controller_cfg=osc_position_cfg,
        )
        steps += 1
        time.sleep(dt)

    current_pos = _current_pos(robot_interface)
    err = target_xyz - current_pos
    if lock_axes == "xy":
        pos_err = float(np.linalg.norm(err[:2]))
    elif lock_axes == "z":
        pos_err = abs(float(err[2]))
    else:
        pos_err = float(np.linalg.norm(err))
    return steps, pos_err


def lift_ee_z(
    robot_interface,
    *,
    lift_m: float,
    osc_position_cfg,
    reset_cfg: FastResetConfig | None = None,
    gripper_cmd: float | None = None,
    logger=None,
) -> tuple[int, float]:
    """Raise EE straight up by ``lift_m`` (xy locked) before any归位 motion.

    Used after a successful insertion: pulling the plug vertically out of the
    socket first avoids the lateral drag that xy-first reset would otherwise
    impose on the still-gripped plug.
    """
    reset_cfg = reset_cfg or FastResetConfig()
    log = _resolve_log(logger)
    if lift_m <= 0.0:
        return 0, 0.0

    _wait_for_state(robot_interface)
    current = _current_pos(robot_interface)
    target = current.copy()
    target[2] = current[2] + float(lift_m)
    log(f"[fast_reset] success lift z +{lift_m*100:.1f}cm before归位 (z {current[2]:.4f}→{target[2]:.4f})")
    return fast_move_to_xyz(
        robot_interface,
        target,
        osc_position_cfg=osc_position_cfg,
        reset_cfg=reset_cfg,
        gripper_cmd=reset_cfg.gripper_closed if gripper_cmd is None else gripper_cmd,
        lock_axes="z",
    )


def fast_approach_to_xyz(
    robot_interface,
    target_xyz: np.ndarray,
    *,
    osc_position_cfg,
    reset_cfg: FastResetConfig | None = None,
    floor_z: float | None = None,
    gripper_cmd: float | None = None,
) -> tuple[int, float]:
    """Safe reset motion: slide xy at current height, then descend z (never below floor_z)."""
    reset_cfg = reset_cfg or FastResetConfig()
    target_xyz = np.asarray(target_xyz, dtype=np.float64).reshape(3).copy()
    if floor_z is not None:
        target_xyz[2] = max(target_xyz[2], floor_z)

    total_steps = 0
    current = _current_pos(robot_interface)
    tx, ty, tz = target_xyz

    if reset_cfg.approach_xy_first:
        xy_err = float(np.linalg.norm([tx - current[0], ty - current[1]]))
        if xy_err > reset_cfg.xy_tol_m:
            hover = np.array([tx, ty, current[2]], dtype=np.float64)
            s, _ = fast_move_to_xyz(
                robot_interface,
                hover,
                osc_position_cfg=osc_position_cfg,
                reset_cfg=reset_cfg,
                gripper_cmd=gripper_cmd,
                lock_axes="xy",
                floor_z=floor_z,
                tol_m=reset_cfg.xy_tol_m,
            )
            total_steps += s
            current = _current_pos(robot_interface)

        if tz < current[2] - reset_cfg.xy_tol_m:
            z_target = np.array([tx, ty, max(tz, floor_z if floor_z is not None else tz)])
            s, pos_err = fast_move_to_xyz(
                robot_interface,
                z_target,
                osc_position_cfg=osc_position_cfg,
                reset_cfg=reset_cfg,
                gripper_cmd=gripper_cmd,
                lock_axes="z",
                floor_z=floor_z,
            )
            return total_steps + s, pos_err
        if abs(tz - current[2]) > reset_cfg.xy_tol_m and tz >= current[2]:
            z_target = np.array([tx, ty, tz], dtype=np.float64)
            s, pos_err = fast_move_to_xyz(
                robot_interface,
                z_target,
                osc_position_cfg=osc_position_cfg,
                reset_cfg=reset_cfg,
                gripper_cmd=gripper_cmd,
                lock_axes="z",
                floor_z=floor_z,
            )
            return total_steps + s, pos_err

    s, pos_err = fast_move_to_xyz(
        robot_interface,
        target_xyz,
        osc_position_cfg=osc_position_cfg,
        reset_cfg=reset_cfg,
        gripper_cmd=gripper_cmd,
        floor_z=floor_z,
    )
    return total_steps + s, pos_err


def reset_to_collection_init(
    robot_interface,
    *,
    gripper,
    cube_cfg: InitCubeConfig,
    osc_position_cfg,
    joint_controller_cfg,
    reset_cfg: FastResetConfig | None = None,
    randomize: bool = True,
    logger=None,
) -> FastResetResult:
    reset_cfg = reset_cfg or FastResetConfig()
    log = _resolve_log(logger)
    rng = np.random.default_rng(cube_cfg.seed)

    cube = InitCubeConfig(
        enabled=randomize and cube_cfg.enabled,
        bottom_center_xyz=cube_cfg.bottom_center_xyz,
        reference_quat=cube_cfg.reference_quat,
        gripper_width=cube_cfg.gripper_width,
        xy_half_range_m=cube_cfg.xy_half_range_m,
        z_range_m=cube_cfg.z_range_m,
        seed=cube_cfg.seed,
    )
    target = sample_init_pose_in_cube(cube, rng)
    bc = np.asarray(cube_cfg.bottom_center_xyz, dtype=np.float64)
    offset = np.asarray(target.ee_pose, dtype=np.float64) - bc

    _wait_for_state(robot_interface)
    pos = _current_pos(robot_interface)
    delta = float(np.linalg.norm(np.asarray(target.ee_pose) - pos))

    if reset_cfg.skip_if_within_m > 0 and delta <= reset_cfg.skip_if_within_m:
        log(f"[fast_reset] skip — already within {delta*100:.1f}cm")
        if gripper is not None and cube_cfg.gripper_width <= 0.001:
            try:
                robot_interface.gripper_control(1.0)
            except Exception:
                pass
        return FastResetResult(
            target_xyz=np.asarray(target.ee_pose, dtype=np.float64),
            offset_xyz=offset,
            steps=0,
            joint_home_used=False,
            motion_skipped=True,
            pos_err_m=delta,
        )

    log(
        f"[fast_reset] bottom_center={np.round(bc, 4).tolist()} "
        f"→ target={np.round(target.ee_pose, 4).tolist()} "
        f"offset(cm)={np.round(offset * 100, 2).tolist()}"
    )

    joint_home = _maybe_joint_home(
        robot_interface,
        np.asarray(target.ee_pose),
        joint_controller_cfg=joint_controller_cfg,
        reset_cfg=reset_cfg,
    )
    if joint_home:
        log("[fast_reset] joint home (large Δ) then xy→z approach")

    floor_z = cube_cfg.floor_z
    steps, pos_err = fast_approach_to_xyz(
        robot_interface,
        np.asarray(target.ee_pose),
        osc_position_cfg=osc_position_cfg,
        reset_cfg=reset_cfg,
        floor_z=floor_z,
    )

    if gripper is not None and cube_cfg.gripper_width <= 0.001:
        try:
            robot_interface.gripper_control(1.0)
        except Exception:
            pass

    log(f"[fast_reset] done steps={steps} pos_err={pos_err*100:.2f}cm")
    return FastResetResult(
        target_xyz=np.asarray(target.ee_pose, dtype=np.float64),
        offset_xyz=offset,
        steps=steps,
        joint_home_used=joint_home,
        motion_skipped=False,
        pos_err_m=pos_err,
    )


def move_reset_pose_fast(
    robot_interface,
    target: ResetPose,
    *,
    osc_position_cfg,
    joint_controller_cfg,
    reset_cfg: FastResetConfig | None = None,
    gripper=None,
    logger=None,
) -> dict:
    """Fast path for demo_reset.move_to_demo_pose (compatible return dict)."""
    from rlt.hardware.deoxys.demo_reset import _current_ee_pose, pose_errors

    reset_cfg = reset_cfg or FastResetConfig()
    log = _resolve_log(logger)

    _wait_for_state(robot_interface)
    pos, quat = _current_ee_pose(robot_interface)
    delta_m = float(np.linalg.norm(pos.reshape(3) - target.ee_pose.reshape(3)))

    if reset_cfg.skip_if_within_m > 0 and delta_m <= reset_cfg.skip_if_within_m:
        width = float(gripper.position) if gripper is not None else target.gripper_width
        pos_err, rot_err = pose_errors(pos, quat, target)
        return {
            "success": True,
            "episode_id": target.episode_id,
            "steps": 0,
            "motion_skipped": True,
            "pos_err_m": pos_err,
            "rot_err_deg": rot_err,
            "gripper_width": width,
            "fast_reset": True,
        }

    log(f"[fast_reset] demo target {target.episode_id} xyz={np.round(target.ee_pose, 3).tolist()}")
    joint_home = _maybe_joint_home(
        robot_interface,
        np.asarray(target.ee_pose),
        joint_controller_cfg=joint_controller_cfg,
        reset_cfg=reset_cfg,
    )
    steps, pos_err = fast_approach_to_xyz(
        robot_interface,
        np.asarray(target.ee_pose),
        osc_position_cfg=osc_position_cfg,
        reset_cfg=reset_cfg,
        floor_z=float(target.ee_pose[2]),
    )
    if gripper is not None and target.gripper_width <= 0.001:
        try:
            robot_interface.gripper_control(1.0)
        except Exception:
            pass

    pos, quat = _current_ee_pose(robot_interface)
    _, rot_err = pose_errors(pos, quat, target)
    width = float(gripper.position) if gripper is not None else target.gripper_width
    return {
        "success": pos_err <= reset_cfg.pos_tol_m * 2,
        "episode_id": target.episode_id,
        "steps": steps,
        "motion_skipped": False,
        "joint_home_used": joint_home,
        "pos_err_m": pos_err,
        "rot_err_deg": rot_err,
        "gripper_width": width,
        "fast_reset": True,
    }
