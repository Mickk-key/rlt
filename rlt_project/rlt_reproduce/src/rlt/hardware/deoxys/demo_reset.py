"""Demo-driven reset: sample critical-phase initial poses from recorded demos.

Motion pipeline (deoxys ``motion_utils`` only — no OSC pose slam):
  1. ``home`` — ``joint`` (teleop joint home, z≈0.61) OR ``hover`` (OSC to hover above target demo, z≈0.25)
  2. Axis-decoupled ``position_only``: **xy at current z** → **vertical z** → final demo xyz

Target pose = ``proprio[0]`` of each collected episode (critical-phase start). This is
**optional** for teleop collection (home reset via SpaceMouse RIGHT is the reliable default).
Orientation from demos is recorded but **not** forced at reset (OSC full-pose caused large jumps).
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rlt.teleop.spacemouse_control import DEFAULT_RESET_JOINTS

POS_TOL_M = 0.01
ROT_TOL_DEG = 5.0
DEFAULT_CONTROL_HZ = 20.0

SegmentMode = str  # "xy" | "z" | "full"


@dataclass(frozen=True)
class PositionSegment:
    """One axis-decoupled position move (OSC cannot hold z while sliding xy in 3D)."""

    target: np.ndarray  # (3,) desired tx, ty, tz
    mode: SegmentMode
    desc: str
    hold_z: float | None = None  # for mode=xy: lock commanded z to this height


@dataclass
class DemoResetSafetyConfig:
    """Conservative motion limits for demo-driven reset."""

    home_joints: list[float] = field(default_factory=lambda: list(DEFAULT_RESET_JOINTS))
    require_home_first: bool = True
    approach_clearance_m: float = 0.05
    min_hover_z_m: float = 0.20
    position_tol_m: float = 0.030
    position_max_rounds: int = 12
    position_min_steps: int = 100
    position_max_steps: int = 400
    trim_orientation: bool = False
    max_pos_delta_m: float = 0.50
    min_target_z_m: float = 0.12
    max_target_z_m: float = 0.80
    joint_reset_timeout: float = 12.0
    post_home_wait_sec: float = 0.4
    gripper_hold_closed: bool = True
    skip_if_within_m: float = 0.0
    home_fallback_delta_m: float = 0.35
    home_mode: str = "joint"  # joint | hover — hover = OSC to hover above target demo (low z)
    direct_move_under_m: float = 0.12  # total Δ below this → one-shot xyz move (not decoupled xy/z)
    fast_reset: bool = False  # use fast_reset.fast_move_to_xyz (no xy/z segments)


@dataclass(frozen=True)
class ResetPose:
    """End-effector reset target (frame 0 of a demo episode)."""

    ee_pose: np.ndarray  # (3,) xyz meters
    quaternion: np.ndarray  # (4,) w,x,y,z
    gripper_width: float
    episode_id: str
    success: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ee_pose": self.ee_pose.tolist(),
            "quaternion": self.quaternion.tolist(),
            "gripper_width": float(self.gripper_width),
            "episode_id": self.episode_id,
            "success": self.success,
        }


def _normalize_quat(quat: Sequence[float]) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def _pose_from_frame_dict(frame: dict, episode_id: str, success: bool | None) -> ResetPose | None:
    ee = frame.get("ee_pose") or frame.get("position") or frame.get("xyz")
    quat = frame.get("quaternion") or frame.get("quat") or frame.get("orientation")
    gripper = frame.get("gripper_width")
    if gripper is None:
        gripper = frame.get("gripper")
    if ee is None or quat is None or gripper is None:
        return None
    ee_arr = np.asarray(ee, dtype=np.float64).reshape(3)
    if np.linalg.norm(ee_arr) < 1e-4:
        return None
    return ResetPose(
        ee_pose=ee_arr.astype(np.float32),
        quaternion=_normalize_quat(quat).astype(np.float32),
        gripper_width=float(gripper),
        episode_id=episode_id,
        success=success,
    )


def _pose_from_proprio(proprio0: np.ndarray, episode_id: str, success: bool | None) -> ResetPose | None:
    p = np.asarray(proprio0, dtype=np.float64).reshape(-1)
    if p.shape[0] < 8:
        return None
    ee = p[:3]
    if np.linalg.norm(ee) < 1e-4:
        return None
    return ResetPose(
        ee_pose=ee.astype(np.float32),
        quaternion=_normalize_quat(p[3:7]).astype(np.float32),
        gripper_width=float(p[7]),
        episode_id=episode_id,
        success=success,
    )


def _parse_json_episode(path: Path, data: dict) -> ResetPose | None:
    episode_id = str(data.get("episode_id") or path.stem)
    success = data.get("success")
    if success is not None:
        success = bool(success)

    if "initial_state" in data:
        return _pose_from_frame_dict(data["initial_state"], episode_id, success)

    frames = data.get("frames") or data.get("trajectory") or data.get("steps")
    if isinstance(frames, list) and frames:
        return _pose_from_frame_dict(frames[0], episode_id, success)

    if all(k in data for k in ("ee_pose", "quaternion", "gripper_width")):
        return _pose_from_frame_dict(data, episode_id, success)

    return None


def _load_pose_from_npz(path: Path) -> ResetPose | None:
    with np.load(path, allow_pickle=False) as data:
        if "proprio" not in data:
            return None
        meta_success = None
        if "metadata_json" in data:
            meta = json.loads(str(data["metadata_json"]))
            meta_success = bool(meta.get("success")) if "success" in meta else None
        return _pose_from_proprio(data["proprio"][0], path.stem, meta_success)


def load_reset_poses(dataset_dir: str | Path) -> list[dict]:
    """Load frame-0 poses from JSON and/or NPZ demos under ``dataset_dir``."""
    root = Path(dataset_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"demo_reset dataset not found: {root}")

    poses: list[ResetPose] = []
    seen: set[str] = set()

    json_files = sorted(root.rglob("*.json"))
    json_files = [p for p in json_files if p.name not in {"manifest.json", "norm_stats.json"}]

    for path in json_files:
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                pose = _parse_json_episode(path, item)
                if pose is not None and pose.episode_id not in seen:
                    poses.append(pose)
                    seen.add(pose.episode_id)
        elif isinstance(data, dict):
            if "episodes" in data and isinstance(data["episodes"], list):
                for item in data["episodes"]:
                    if not isinstance(item, dict):
                        continue
                    ep_id = str(item.get("episode_id") or item.get("file", "unknown"))
                    if "initial_state" in item:
                        pose = _pose_from_frame_dict(item["initial_state"], ep_id, item.get("success"))
                    elif "ee_pose" in item:
                        pose = _pose_from_frame_dict(item, ep_id, item.get("success"))
                    else:
                        pose = None
                    if pose is not None and pose.episode_id not in seen:
                        poses.append(pose)
                        seen.add(pose.episode_id)
            else:
                pose = _parse_json_episode(path, data)
                if pose is not None and pose.episode_id not in seen:
                    poses.append(pose)
                    seen.add(pose.episode_id)

    for path in sorted(root.rglob("*.npz")):
        if path.stem in seen:
            continue
        pose = _load_pose_from_npz(path)
        if pose is not None:
            poses.append(pose)
            seen.add(pose.episode_id)

    if not poses:
        raise RuntimeError(f"No valid reset poses found under {root}")

    return [p.as_dict() for p in poses]


class DemoResetSampler:
    """Random sampler over demo initial poses."""

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        seed: int | None = None,
        safety: DemoResetSafetyConfig | None = None,
        pin_episode_id: str | None = None,
    ):
        safety = safety or DemoResetSafetyConfig()
        raw_poses = load_reset_poses(dataset_dir)
        self._poses: list[ResetPose] = []
        for p in raw_poses:
            pose = ResetPose(
                ee_pose=np.asarray(p["ee_pose"], dtype=np.float32),
                quaternion=_normalize_quat(p["quaternion"]).astype(np.float32),
                gripper_width=float(p["gripper_width"]),
                episode_id=str(p["episode_id"]),
                success=p.get("success"),
            )
            try:
                validate_reset_pose(pose, None, safety)
                self._poses.append(pose)
            except ValueError as exc:
                print(f"[demo_reset] skip {pose.episode_id}: {exc}")

        if not self._poses:
            raise RuntimeError("No demo poses passed safety filters")

        self._rng = random.Random(seed)
        self._pin_episode_id = pin_episode_id.strip() if pin_episode_id else None
        if self._pin_episode_id:
            self._pin_index = next(
                (i for i, p in enumerate(self._poses) if p.episode_id == self._pin_episode_id),
                None,
            )
            if self._pin_index is None:
                raise ValueError(
                    f"demo_reset_pin_episode={self._pin_episode_id!r} not in pool "
                    f"({len(self._poses)} poses)"
                )
            print(f"[demo_reset] pinned episode {self._pin_episode_id} (index={self._pin_index})")
        else:
            self._pin_index = None
        self._last_index: int | None = None
        self._last_pose: ResetPose | None = None

    def __len__(self) -> int:
        return len(self._poses)

    def sample_reset_pose(self) -> dict:
        if self._pin_index is not None:
            idx = self._pin_index
        else:
            idx = self._rng.randrange(len(self._poses))
        self._last_index = idx
        self._last_pose = self._poses[idx]
        out = self._last_pose.as_dict()
        out["demo_index"] = idx
        return out

    @property
    def last_demo_index(self) -> int | None:
        return self._last_index

    @property
    def last_pose(self) -> ResetPose | None:
        return self._last_pose


def sample_reset_pose(dataset_dir: str | Path, *, seed: int | None = None) -> dict:
    return DemoResetSampler(dataset_dir, seed=seed).sample_reset_pose()


def quat_geodesic_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = _normalize_quat(q1)
    q2 = _normalize_quat(q2)
    dot = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def pose_errors(current_pos: np.ndarray, current_quat: np.ndarray, target: ResetPose) -> tuple[float, float]:
    pos_err = float(np.linalg.norm(current_pos.reshape(3) - target.ee_pose.reshape(3)))
    rot_err = quat_geodesic_deg(current_quat, target.quaternion)
    return pos_err, rot_err


def validate_reset_pose(
    target: ResetPose,
    current_pos: np.ndarray | None,
    safety: DemoResetSafetyConfig,
) -> None:
    z = float(target.ee_pose[2])
    if z < safety.min_target_z_m or z > safety.max_target_z_m:
        raise ValueError(f"target z={z:.3f}m outside [{safety.min_target_z_m}, {safety.max_target_z_m}]")
    if current_pos is not None:
        delta = float(np.linalg.norm(np.asarray(current_pos).reshape(3) - target.ee_pose.reshape(3)))
        if delta > safety.max_pos_delta_m:
            raise ValueError(
                f"cartesian delta {delta:.3f}m exceeds max {safety.max_pos_delta_m}m — use home first or fix demo"
            )


def _wait_for_state(robot_interface, timeout: float = 8.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if robot_interface._state_buffer:
            if hasattr(robot_interface, "check_nonzero_configuration"):
                if robot_interface.check_nonzero_configuration():
                    return
            else:
                return
        time.sleep(0.02)
    raise TimeoutError("Timed out waiting for robot state before demo reset")


def _current_ee_pos(robot_interface) -> np.ndarray:
    _, pos = robot_interface.last_eef_rot_and_pos
    return np.asarray(pos, dtype=np.float64).reshape(3)


def _current_ee_pose(robot_interface) -> tuple[np.ndarray, np.ndarray]:
    from rlt.hardware.deoxys_arm import o_t_ee_to_pose

    st = robot_interface._state_buffer[-1]
    pos, quat = o_t_ee_to_pose(st.O_T_EE)
    return np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)


def _gripper_action(safety: DemoResetSafetyConfig) -> float:
    return 1.0 if safety.gripper_hold_closed else -1.0


def move_home_joints(
    robot_interface,
    safety: DemoResetSafetyConfig,
    *,
    joint_controller_cfg,
) -> None:
    """Safe joint-space home via deoxys ``reset_joints_to`` (same as reset_robot.sh)."""
    from deoxys.experimental.motion_utils import reset_joints_to

    print(f"[demo_reset] phase 0/2: joint home ({len(safety.home_joints)} joints)")
    reset_joints_to(
        robot_interface,
        list(safety.home_joints),
        controller_cfg=joint_controller_cfg,
        timeout=safety.joint_reset_timeout,
        gripper_open=not safety.gripper_hold_closed,
    )
    if safety.post_home_wait_sec > 0:
        time.sleep(safety.post_home_wait_sec)


def _hover_xyz(target: ResetPose, safety: DemoResetSafetyConfig) -> np.ndarray:
    tx, ty, tz = [float(v) for v in target.ee_pose]
    hz = max(safety.min_hover_z_m, tz + safety.approach_clearance_m)
    return np.array([tx, ty, hz], dtype=np.float64)


def move_hover_home(
    robot_interface,
    target: ResetPose,
    safety: DemoResetSafetyConfig,
    *,
    osc_position_cfg,
) -> int:
    """OSC position-only to hover above *this* demo target (z≈0.25), not joint home at z≈0.61."""
    hover_pose = ResetPose(
        ee_pose=_hover_xyz(target, safety).astype(np.float32),
        quaternion=target.quaternion,
        gripper_width=target.gripper_width,
        episode_id=f"{target.episode_id}_hover",
    )
    pos = _current_ee_pos(robot_interface)
    hz = hover_pose.ee_pose[2]
    print(
        f"[demo_reset] phase 0/2: hover home above {target.episode_id} "
        f"xyz={np.round(hover_pose.ee_pose, 3).tolist()}"
    )
    waypoints = _position_waypoints(pos, hover_pose, safety)
    total = 0
    for i, segment in enumerate(waypoints):
        total += _position_segment_until(
            robot_interface,
            segment,
            osc_position_cfg=osc_position_cfg,
            safety=safety,
            label=f"hover{i+1}/{len(waypoints)} {segment.desc}",
        )
    return total


def _position_waypoints(
    current_pos: np.ndarray,
    target: ResetPose,
    safety: DemoResetSafetyConfig,
) -> list[PositionSegment]:
    """xy at home height (z locked) → vertical descend → final z."""
    tx, ty, tz = [float(v) for v in target.ee_pose]
    hover_z = max(safety.min_hover_z_m, tz + safety.approach_clearance_m)
    cx, cy, cz = [float(v) for v in current_pos.reshape(3)]
    segments: list[PositionSegment] = []

    total_dist = float(np.linalg.norm([tx - cx, ty - cy, tz - cz]))
    if safety.direct_move_under_m > 0 and total_dist <= safety.direct_move_under_m:
        return [
            PositionSegment(
                target=np.array([tx, ty, tz]),
                mode="full",
                desc=f"direct Δ={total_dist*100:.1f}cm",
            )
        ]

    xy_dist = float(np.linalg.norm([tx - cx, ty - cy]))
    near_height = abs(cz - tz) < 0.15

    if near_height:
        if xy_dist > 0.008:
            segments.append(
                PositionSegment(
                    target=np.array([tx, ty, tz]),
                    mode="full",
                    desc=f"near-height direct xy Δ={xy_dist*100:.1f}cm",
                )
            )
        elif abs(cz - tz) > 0.008:
            segments.append(
                PositionSegment(
                    target=np.array([tx, ty, tz]),
                    mode="z",
                    desc=f"target z={tz:.3f}",
                )
            )
        if not segments:
            segments.append(
                PositionSegment(
                    target=np.array([tx, ty, tz]),
                    mode="full",
                    desc="near-height trim",
                )
            )
        return segments

    if xy_dist > 0.008:
        segments.append(
            PositionSegment(
                target=np.array([tx, ty, cz]),
                mode="xy",
                desc=f"xy at z={cz:.3f}",
                hold_z=cz,
            )
        )
    if abs(cz - hover_z) > 0.008:
        segments.append(
            PositionSegment(
                target=np.array([tx, ty, hover_z]),
                mode="z",
                desc=f"descend hover z={hover_z:.3f}",
            )
        )
    if abs(hover_z - tz) > 0.008 or not segments:
        segments.append(
            PositionSegment(
                target=np.array([tx, ty, tz]),
                mode="z",
                desc=f"target z={tz:.3f}",
            )
        )
    return segments


def _segment_error(pos: np.ndarray, segment: PositionSegment) -> float:
    t = segment.target.reshape(3)
    if segment.mode == "xy":
        return float(np.linalg.norm(t[:2] - pos[:2]))
    if segment.mode == "z":
        return abs(float(t[2] - pos[2]))
    return float(np.linalg.norm(t - pos))


def _segment_command(pos: np.ndarray, segment: PositionSegment) -> np.ndarray:
    t = segment.target.reshape(3)
    if segment.mode == "xy":
        lock_z = segment.hold_z if segment.hold_z is not None else float(pos[2])
        return np.array([t[0], t[1], lock_z], dtype=np.float64)
    if segment.mode == "z":
        return np.array([pos[0], pos[1], t[2]], dtype=np.float64)
    return t.copy()


def _incremental_position_segment(
    robot_interface,
    segment: PositionSegment,
    *,
    osc_position_cfg,
    safety: DemoResetSafetyConfig,
    label: str = "",
    accept_tol_m: float | None = None,
) -> int:
    """Per-step OSC_POSITION deltas (fallback when batch ``move_to`` stalls)."""
    trans_scale = float(
        getattr(osc_position_cfg.action_scale, "translation", None)
        or osc_position_cfg.get("action_scale", {}).get("translation", 0.05)
    )
    grasp = _gripper_action(safety)
    total_steps = 0
    accept = accept_tol_m if accept_tol_m is not None else safety.position_tol_m

    for attempt in range(safety.position_max_rounds):
        pos = _current_ee_pos(robot_interface)
        err = _segment_error(pos, segment)
        if err < safety.position_tol_m:
            print(
                f"[demo_reset]   {label} ok (incremental) {segment.mode} err={err*100:.2f}cm "
                f"ee={np.round(pos, 3).tolist()} rounds={attempt}"
            )
            return total_steps

        cmd = _segment_command(pos, segment)
        delta = cmd - pos
        if segment.mode == "xy":
            delta[2] = 0.0
        elif segment.mode == "z":
            delta[0] = 0.0
            delta[1] = 0.0

        dist = float(np.linalg.norm(delta))
        if dist < 1e-5:
            break

        steps = int(np.clip(dist / trans_scale + 20, 30, safety.position_max_steps))
        for _ in range(steps):
            pos = _current_ee_pos(robot_interface)
            err = _segment_error(pos, segment)
            if err < safety.position_tol_m:
                print(
                    f"[demo_reset]   {label} ok (incremental) {segment.mode} err={err*100:.2f}cm "
                    f"ee={np.round(pos, 3).tolist()}"
                )
                return total_steps

            cmd = _segment_command(pos, segment)
            delta = cmd - pos
            if segment.mode == "xy":
                delta[2] = 0.0
            elif segment.mode == "z":
                delta[0] = 0.0
                delta[1] = 0.0
            dist = float(np.linalg.norm(delta))
            if dist < 1e-5:
                break

            direction = delta / dist
            step_m = min(dist, trans_scale * 0.95)
            action = np.zeros(7, dtype=np.float64)
            action[:3] = direction * (step_m / trans_scale)
            action[6] = grasp
            robot_interface.control(
                controller_type="OSC_POSITION",
                action=action,
                controller_cfg=osc_position_cfg,
            )
            total_steps += 1

    pos = _current_ee_pos(robot_interface)
    err = _segment_error(pos, segment)
    if err <= accept:
        print(
            f"[demo_reset]   {label} soft-accept {segment.mode} err={err*100:.2f}cm "
            f"(accept<={accept*100:.1f}cm) ee={np.round(pos, 3).tolist()}"
        )
        return total_steps
    raise TimeoutError(
        f"{label} timed out {segment.mode} err={err*100:.2f}cm pos={np.round(pos, 3).tolist()} "
        f"target={np.round(segment.target.reshape(3), 3).tolist()}"
    )


def _position_segment_until(
    robot_interface,
    segment: PositionSegment,
    *,
    osc_position_cfg,
    safety: DemoResetSafetyConfig,
    label: str = "",
    accept_tol_m: float | None = None,
) -> int:
    """Axis-decoupled ``position_only_gripper_move_to`` with retry until segment tol."""
    from deoxys.experimental.motion_utils import position_only_gripper_move_to

    grasp = safety.gripper_hold_closed
    total_steps = 0
    accept = accept_tol_m if accept_tol_m is not None else safety.position_tol_m

    for attempt in range(safety.position_max_rounds):
        pos = _current_ee_pos(robot_interface)
        err = _segment_error(pos, segment)
        if err < safety.position_tol_m:
            err_tag = f"{segment.mode} err={err*100:.2f}cm"
            print(
                f"[demo_reset]   {label} ok {err_tag} "
                f"ee={np.round(pos, 3).tolist()} rounds={attempt}"
            )
            return total_steps

        cmd = _segment_command(pos, segment).reshape(3, 1)
        dist = float(np.linalg.norm(cmd.reshape(3) - pos))
        min_steps = max(30, safety.position_min_steps // 2) if dist < 0.06 else safety.position_min_steps
        num_steps = int(
            np.clip(dist / 0.0015 + min_steps, min_steps, safety.position_max_steps)
        )
        position_only_gripper_move_to(
            robot_interface,
            cmd,
            num_steps=num_steps,
            controller_cfg=osc_position_cfg,
            grasp=grasp,
        )
        total_steps += num_steps

    pos = _current_ee_pos(robot_interface)
    err = _segment_error(pos, segment)
    if err <= accept:
        print(
            f"[demo_reset]   {label} soft-accept {segment.mode} err={err*100:.2f}cm "
            f"(accept<={accept*100:.1f}cm) ee={np.round(pos, 3).tolist()}"
        )
        return total_steps

    print(f"[demo_reset]   {label} batch stalled err={err*100:.2f}cm — trying incremental")
    return _incremental_position_segment(
        robot_interface,
        segment,
        osc_position_cfg=osc_position_cfg,
        safety=safety,
        label=label,
        accept_tol_m=accept,
    )


def _set_gripper_width(robot_interface, gripper, target_width: float, *, tol: float = 0.004, timeout: float = 3.0) -> None:
    if gripper is None or not getattr(robot_interface, "has_gripper", False):
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        width = float(gripper.position)
        if abs(width - target_width) <= tol:
            return
        cmd = 1.0 if target_width < width else -1.0
        robot_interface.gripper_control(cmd)
        time.sleep(0.05)


def move_to_demo_pose(
    robot_interface,
    target: ResetPose | dict,
    *,
    controller_cfg,
    controller_type: str = "OSC_POSE",
    gripper=None,
    pos_tol_m: float = POS_TOL_M,
    rot_tol_deg: float = ROT_TOL_DEG,
    gripper_tol_m: float = 0.004,
    control_hz: float = DEFAULT_CONTROL_HZ,
    safety: DemoResetSafetyConfig | None = None,
    joint_controller_cfg=None,
    osc_position_cfg=None,
    logger=None,
) -> dict:
    """Home joints → position_only to demo xyz (proprio[0] from NPZ). Orientation not forced."""
    del controller_type, controller_cfg, control_hz

    if isinstance(target, dict):
        target = ResetPose(
            ee_pose=np.asarray(target["ee_pose"], dtype=np.float32),
            quaternion=_normalize_quat(target["quaternion"]).astype(np.float32),
            gripper_width=float(target["gripper_width"]),
            episode_id=str(target.get("episode_id", "unknown")),
            success=target.get("success"),
        )

    safety = safety or DemoResetSafetyConfig()
    log = logger.info if logger is not None else print

    _wait_for_state(robot_interface)

    if joint_controller_cfg is None or osc_position_cfg is None:
        raise ValueError("joint_controller_cfg and osc_position_cfg are required for demo reset")

    if getattr(safety, "fast_reset", False):
        from rlt.hardware.deoxys.fast_reset import FastResetConfig, move_reset_pose_fast

        fast_cfg = FastResetConfig(
            control_hz=float(control_hz) if control_hz else DEFAULT_CONTROL_HZ,
            pos_tol_m=pos_tol_m,
            home_joints=list(safety.home_joints),
            joint_home_if_delta_above_m=float(safety.home_fallback_delta_m),
            skip_if_within_m=float(safety.skip_if_within_m),
        )
        return move_reset_pose_fast(
            robot_interface,
            target,
            osc_position_cfg=osc_position_cfg,
            joint_controller_cfg=joint_controller_cfg,
            reset_cfg=fast_cfg,
            gripper=gripper,
            logger=logger,
        )

    total_steps = 0

    pos, quat = _current_ee_pose(robot_interface)
    delta_m = float(np.linalg.norm(pos.reshape(3) - target.ee_pose.reshape(3)))

    if safety.skip_if_within_m > 0 and delta_m <= safety.skip_if_within_m:
        log(
            f"[demo_reset] skip motion — already within {delta_m*100:.1f}cm "
            f"(tol={safety.skip_if_within_m*100:.1f}cm)"
        )
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
        }

    need_home = safety.require_home_first
    if (
        not need_home
        and safety.home_fallback_delta_m > 0
        and delta_m > safety.home_fallback_delta_m
    ):
        log(
            f"[demo_reset] Δ={delta_m*100:.1f}cm > fallback {safety.home_fallback_delta_m*100:.0f}cm "
            f"— {safety.home_mode} home first"
        )
        need_home = True

    if need_home:
        if safety.home_mode == "hover":
            total_steps += move_hover_home(
                robot_interface,
                target,
                safety,
                osc_position_cfg=osc_position_cfg,
            )
        else:
            move_home_joints(robot_interface, safety, joint_controller_cfg=joint_controller_cfg)
        pos, quat = _current_ee_pose(robot_interface)
        delta_m = float(np.linalg.norm(pos.reshape(3) - target.ee_pose.reshape(3)))

    validate_reset_pose(target, pos, safety)
    log(
        f"[demo_reset] target {target.episode_id} from demo proprio[0] "
        f"xyz={np.round(target.ee_pose, 3).tolist()}"
    )
    log(f"[demo_reset]   Δ ee={delta_m*100:.1f}cm (current z≈{pos[2]:.2f} → demo z≈{target.ee_pose[2]:.2f})")

    waypoints = _position_waypoints(pos, target, safety)
    log(f"[demo_reset] phase 1/2: {len(waypoints)} position segments (position_only)")

    def _run_segments() -> int:
        steps = 0
        for i, segment in enumerate(waypoints):
            steps += _position_segment_until(
                robot_interface,
                segment,
                osc_position_cfg=osc_position_cfg,
                safety=safety,
                label=f"pos{i+1}/{len(waypoints)} {segment.desc}",
                accept_tol_m=pos_tol_m,
            )
        return steps

    home_retried = False
    try:
        total_steps += _run_segments()
    except TimeoutError as exc:
        pos, quat = _current_ee_pose(robot_interface)
        pos_err, _ = pose_errors(pos, quat, target)
        if pos_err <= pos_tol_m:
            log(
                f"[demo_reset] soft-accept final pos err={pos_err*100:.2f}cm "
                f"(limit={pos_tol_m*100:.1f}cm) after segment stall"
            )
        elif not need_home and not home_retried:
            log(f"[demo_reset] segment failed ({exc}) — joint home once and retry")
            move_home_joints(robot_interface, safety, joint_controller_cfg=joint_controller_cfg)
            pos, quat = _current_ee_pose(robot_interface)
            waypoints = _position_waypoints(pos, target, safety)
            log(f"[demo_reset] retry: {len(waypoints)} position segments after home")
            home_retried = True
            total_steps += _run_segments()
        else:
            raise

    log("[demo_reset] phase 2/2: done (xyz only; orientation from teleop/home is kept)")

    width = float(gripper.position) if gripper is not None else target.gripper_width
    if safety.gripper_hold_closed:
        log(f"[demo_reset] gripper held closed (width={width:.4f}m)")
    else:
        log("[demo_reset] gripper adjust")
        _set_gripper_width(robot_interface, gripper, target.gripper_width, tol=gripper_tol_m)
        width = float(gripper.position) if gripper is not None else target.gripper_width

    pos, quat = _current_ee_pose(robot_interface)
    pos_err, rot_err = pose_errors(pos, quat, target)

    if pos_err >= pos_tol_m:
        raise TimeoutError(
            f"demo reset finished but pos out of tol for {target.episode_id}: "
            f"pos={pos_err*100:.2f}cm (limit={pos_tol_m*100:.1f}cm) rot={rot_err:.1f}deg (not enforced)"
        )
    if safety.trim_orientation and rot_err >= rot_tol_deg:
        raise TimeoutError(
            f"demo reset orientation out of tol for {target.episode_id}: rot={rot_err:.2f}deg"
        )

    log(
        f"[demo_reset] reached {target.episode_id} "
        f"(pos={pos_err*100:.2f}cm rot={rot_err:.2f}deg gripper={width:.4f}m steps≈{total_steps})"
    )
    return {
        "success": True,
        "episode_id": target.episode_id,
        "steps": total_steps,
        "pos_err_m": pos_err,
        "rot_err_deg": rot_err,
        "gripper_width": width,
    }


def _gripper_hold_closed_from_yaml(dc: dict) -> bool:
    if "demo_reset_gripper_hold_closed" in dc:
        return bool(dc["demo_reset_gripper_hold_closed"])
    if "demo_reset_gripper_open" in dc:
        return not bool(dc["demo_reset_gripper_open"])
    return True


def safety_config_from_yaml(dc: dict) -> DemoResetSafetyConfig:
    home = list(
        dc.get("demo_reset_home_joints")
        or dc.get("reset_joint_positions")
        or DEFAULT_RESET_JOINTS
    )
    home_mode = str(dc.get("demo_reset_home_mode", "joint")).lower()
    if home_mode not in ("joint", "hover"):
        raise ValueError(f"demo_reset_home_mode must be joint|hover, got {home_mode!r}")
    return DemoResetSafetyConfig(
        home_joints=home,
        require_home_first=bool(dc.get("demo_reset_require_home_first", True)),
        approach_clearance_m=float(dc.get("demo_reset_approach_clearance_m", 0.05)),
        min_hover_z_m=float(dc.get("demo_reset_min_hover_z_m", 0.20)),
        position_tol_m=float(dc.get("demo_reset_position_tol_m", 0.030)),
        position_max_rounds=int(dc.get("demo_reset_position_max_rounds", 12)),
        position_min_steps=int(dc.get("demo_reset_position_min_steps", 100)),
        position_max_steps=int(dc.get("demo_reset_position_max_steps", 400)),
        trim_orientation=bool(dc.get("demo_reset_trim_orientation", False)),
        max_pos_delta_m=float(dc.get("demo_reset_max_pos_delta_m", 0.50)),
        min_target_z_m=float(dc.get("demo_reset_min_target_z_m", 0.12)),
        max_target_z_m=float(dc.get("demo_reset_max_target_z_m", 0.80)),
        joint_reset_timeout=float(dc.get("demo_reset_joint_timeout", 12.0)),
        post_home_wait_sec=float(dc.get("demo_reset_post_home_wait_sec", 0.4)),
        gripper_hold_closed=_gripper_hold_closed_from_yaml(dc),
        skip_if_within_m=float(dc.get("demo_reset_skip_if_within_m", 0.0)),
        home_fallback_delta_m=float(dc.get("demo_reset_home_fallback_delta_m", 0.35)),
        home_mode=home_mode,
        direct_move_under_m=float(dc.get("demo_reset_direct_move_under_m", 0.12)),
        fast_reset=bool(dc.get("demo_reset_fast_reset", dc.get("fast_reset", False))),
    )
