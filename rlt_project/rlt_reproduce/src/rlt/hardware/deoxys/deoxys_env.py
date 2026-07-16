"""Deoxys FR3 environment with optional demo-driven reset."""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from rlt.hardware.deoxys.demo_reset import (
    DemoResetSampler,
    ResetPose,
    move_to_demo_pose,
    safety_config_from_yaml,
)
from rlt.hardware.deoxys.collection_reset import collection_reset_settings, resolve_reset_yaml
from rlt.hardware.deoxys.fast_reset import InitCubeConfig
from rlt.hardware.workspace_reset import reset_random_workspace
from rlt.hardware.deoxys_arm import DummyArmBackend, o_t_ee_to_pose
from rlt.hardware.franka.gripper import FrankaConfig, FrankaGripperAdapter
from rlt.teleop.spacemouse_control import apply_gripper_latch
from rlt.util.deoxys_paths import (
    default_osc_controller_cfg_name,
    resolve_controller_cfg_path,
    resolve_demo_reset_path,
    smq_root_from_rlt,
)

logger = logging.getLogger(__name__)


@dataclass
class DeoxysEnvConfig:
    backend: str = "dummy"
    deoxys_root: str = ""
    deoxys_config: str = ""
    control_hz: float = 20.0
    controller_type: str = "OSC_POSE"
    controller_cfg_name: str = "osc-position-controller.yml"
    action_scale: tuple[float, float, float] = (0.05, 0.02, 1.0)
    proprio_dim: int = 8
    action_dim: int = 7
    gripper_config: str = "configs/franka/franka_hand.yaml"
    chunk_length: int = 10
    use_demo_reset: bool = False
    demo_reset_path: str = ""
    demo_reset_seed: int | None = None
    demo_reset_pos_tol_m: float = 0.01
    demo_reset_rot_tol_deg: float = 5.0
    reset_joint_positions: list[float] | None = None
    demo_reset_yaml: dict = field(default_factory=dict)
    joint_controller_cfg_name: str = "joint-position-controller.yml"
    gripper_latch: bool = True
    demo_reset_gripper_hold_closed: bool = True
    demo_reset_pin_episode: str | None = None
    use_workspace_reset: bool = False
    workspace_randomization: InitCubeConfig | None = None
    sft_reset_yaml: dict = field(default_factory=dict)
    reset_raw: dict = field(default_factory=dict)
    # GPU/VLA actions are physical EE deltas (m, rad) like NPZ; teleop uses normalized spacemouse units.
    action_is_physical: bool = False
    # --- Hard robot safety limits (applied to BOTH reference and policy modes) ---
    # Bounds the physical per-step EE delta the OSC controller will realize, so the
    # arm never receives an unconstrained actor command (RLT paper: action-space bounding).
    safety_enabled: bool = True
    max_trans_delta_m: float = 0.02
    max_rot_delta_rad: float = 0.1
    gripper_min: float = -1.0
    gripper_max: float = 1.0


class DeoxysEnv:
    """Franka FR3 + Franka Hand via deoxys, with demo-driven episode reset."""

    def __init__(self, cfg: DeoxysEnvConfig):
        self.cfg = cfg
        self._iface = None
        self._gripper: FrankaGripperAdapter | None = None
        self._controller_cfg = None
        self._joint_controller_cfg = None
        self._osc_position_cfg = None
        self._demo_safety = None
        self._step = 0
        self._pose_ready = True
        self._demo_sampler: DemoResetSampler | None = None
        self._last_reset_info: dict = {}
        self._gripper_latched = False
        self._workspace_cfg = cfg.workspace_randomization

        if cfg.use_demo_reset and cfg.demo_reset_path:
            self._demo_safety = safety_config_from_yaml(
                {"reset_joint_positions": cfg.reset_joint_positions, **cfg.demo_reset_yaml}
            )
            self._demo_sampler = DemoResetSampler(
                cfg.demo_reset_path,
                seed=cfg.demo_reset_seed,
                safety=self._demo_safety,
                pin_episode_id=cfg.demo_reset_pin_episode,
            )

        if cfg.backend == "deoxys":
            self._connect_deoxys()
        else:
            self._arm = DummyArmBackend()

    def _connect_deoxys(self, *, control_hz: float | None = None) -> None:
        cfg = self.cfg
        root = Path(cfg.deoxys_root).resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from deoxys import config_root
        from deoxys.franka_interface import FrankaInterface
        from deoxys.utils import YamlConfig
        from deoxys.utils.config_utils import get_default_controller_config

        interface_path = cfg.deoxys_config
        if not interface_path.startswith("/"):
            candidate = Path(config_root) / Path(interface_path).name
            interface_path = str(candidate if candidate.is_file() else interface_path)

        hz = control_hz if control_hz is not None else cfg.control_hz
        self._iface = FrankaInterface(
            interface_path,
            control_freq=hz,
            has_gripper=True,
            automatic_gripper_reset=False,
        )
        self._controller_cfg = get_default_controller_config(cfg.controller_type)
        if cfg.controller_cfg_name:
            cfg_path = resolve_controller_cfg_path(
                cfg.controller_cfg_name,
                smq_root=smq_root_from_rlt(),
                deoxys_config_root=config_root,
            )
            user_cfg = YamlConfig(str(cfg_path)).as_easydict()
            self._controller_cfg = user_cfg

        joint_cfg_path = resolve_controller_cfg_path(
            cfg.joint_controller_cfg_name,
            smq_root=smq_root_from_rlt(),
            deoxys_config_root=config_root,
        )
        self._joint_controller_cfg = YamlConfig(str(joint_cfg_path)).as_easydict()
        osc_pos_path = resolve_controller_cfg_path(
            "configs/deoxys/osc-position-controller.yml",
            smq_root=smq_root_from_rlt(),
            deoxys_config_root=config_root,
        )
        self._osc_position_cfg = YamlConfig(str(osc_pos_path)).as_easydict()
        gripper_cfg = FrankaConfig.from_yaml(Path(cfg.gripper_config))
        self._gripper = FrankaGripperAdapter(self._iface, gripper_cfg)
        time.sleep(0.5)

    def suspend_deoxys_client(self) -> None:
        """Close FrankaInterface so reset_to_init.sh can own ZMQ port 5555."""
        import gc

        if self._gripper is not None:
            try:
                self._gripper.cleanup()
            except Exception:
                pass
            self._gripper = None
        if self._iface is not None:
            try:
                self._iface.close()
            except Exception:
                pass
            self._iface = None
        gc.collect()
        time.sleep(0.5)

    def resume_deoxys_client(self) -> None:
        """Reconnect after external reset subprocess (RL control at robot.control_hz)."""
        if self.cfg.backend != "deoxys":
            return
        if self._iface is not None:
            return
        self._connect_deoxys(control_hz=self.cfg.control_hz)

    @property
    def pose_ready(self) -> bool:
        return self._pose_ready

    @property
    def last_reset_info(self) -> dict:
        return dict(self._last_reset_info)

    def get_proprio(self) -> np.ndarray:
        if self._iface is None:
            arm = self._arm.get_state()
            width = np.array([0.04], dtype=np.float32)
            return np.concatenate([arm.ee_pose, width]).astype(np.float32)

        st = self._iface._state_buffer[-1]
        pos, quat = o_t_ee_to_pose(st.O_T_EE)
        width = np.array([self._gripper.position], dtype=np.float32)
        return np.concatenate([pos, quat, width]).astype(np.float32)

    def _controller_action_scales(self) -> tuple[float, float]:
        if self._controller_cfg is None:
            return 0.05, 1.0
        scale = self._controller_cfg.action_scale
        if hasattr(scale, "translation"):
            return float(scale.translation), float(scale.rotation)
        if isinstance(scale, dict):
            return float(scale.get("translation", 0.05)), float(scale.get("rotation", 1.0))
        return 0.05, 1.0

    def _apply_safety_limits(
        self,
        cmd: np.ndarray,
        *,
        ctrl_trans: float,
        ctrl_rot: float,
    ) -> tuple[np.ndarray, dict]:
        """Hard safety clamp on the controller command before it reaches the arm.

        Applies to BOTH reference and policy modes so the robot never receives an
        unconstrained actor output (RLT paper relies on action-space bounding).

        The OSC controller realizes a physical delta ``phys = cmd * ctrl_scale``.
        The translation (``cmd[:3]``) and axis-angle rotation (``cmd[3:6]``) deltas
        are clamped by *magnitude* (vector norm), preserving direction, so the
        physical translation stays within ``max_trans_delta_m`` (m) and the physical
        rotation angle (= axis-angle norm) within ``max_rot_delta_rad`` (rad). This
        matches the robot action space: ``cmd[3:6]`` is the OSC_POSE rotation-vector
        delta in radians (NOT a quaternion and NOT degrees). Non-finite commands
        hold the pose (zero delta) rather than sending a dangerous command.
        """
        cmd = np.asarray(cmd, dtype=np.float32).copy()
        raw_norm = float(np.linalg.norm(cmd))
        n = cmd.shape[0]

        # Physical per-step deviations actually realized by the controller.
        raw_trans_dev_m = float(np.linalg.norm(cmd[:3])) * ctrl_trans if n >= 3 else 0.0
        raw_rot_dev_rad = float(np.linalg.norm(cmd[3:6])) * ctrl_rot if n >= 6 else 0.0

        if not self.cfg.safety_enabled:
            return cmd, {
                "safety_enabled": False,
                "action_raw_norm": raw_norm,
                "action_clipped_norm": raw_norm,
                "action_clipped": False,
                "action_nan_inf": False,
                "trans_dev_m": raw_trans_dev_m,
                "rot_dev_rad": raw_rot_dev_rad,
            }

        if not np.all(np.isfinite(cmd)):
            logger.error(
                "[safety] non-finite action %s — holding pose (zero delta)", cmd.tolist()
            )
            hold = np.zeros_like(cmd)
            return hold, {
                "safety_enabled": True,
                "action_nan_inf": True,
                "action_raw_norm": raw_norm,
                "action_clipped_norm": 0.0,
                "action_clipped": True,
                "trans_dev_m": raw_trans_dev_m,
                "rot_dev_rad": raw_rot_dev_rad,
            }

        # Caps expressed in controller-command units (phys = cmd * ctrl_scale).
        trans_cap = self.cfg.max_trans_delta_m / max(ctrl_trans, 1e-8)
        rot_cap = self.cfg.max_rot_delta_rad / max(ctrl_rot, 1e-8)
        clipped = False
        if n >= 3:
            cmd[:3], t_clip = self._clip_vec_norm(cmd[:3], trans_cap)
            clipped = clipped or t_clip
        if n >= 6:
            cmd[3:6], r_clip = self._clip_vec_norm(cmd[3:6], rot_cap)
            clipped = clipped or r_clip
        if n >= 7:
            g0 = float(cmd[6])
            cmd[6] = float(np.clip(g0, self.cfg.gripper_min, self.cfg.gripper_max))
            clipped = clipped or (cmd[6] != g0)

        clipped_norm = float(np.linalg.norm(cmd))
        if clipped:
            logger.warning(
                "[safety] clamp applied: raw_norm=%.4f -> clipped_norm=%.4f | "
                "trans_dev=%.4fm (cap %.3f) rot_dev=%.4frad (cap %.3f)",
                raw_norm,
                clipped_norm,
                raw_trans_dev_m,
                self.cfg.max_trans_delta_m,
                raw_rot_dev_rad,
                self.cfg.max_rot_delta_rad,
            )
        return cmd, {
            "safety_enabled": True,
            "action_nan_inf": False,
            "action_raw_norm": raw_norm,
            "action_clipped_norm": clipped_norm,
            "action_clipped": bool(clipped),
            "trans_dev_m": raw_trans_dev_m,
            "rot_dev_rad": raw_rot_dev_rad,
        }

    @staticmethod
    def _clip_vec_norm(vec: np.ndarray, cap: float) -> tuple[np.ndarray, bool]:
        """Scale a vector down to ``cap`` magnitude, preserving direction."""
        norm = float(np.linalg.norm(vec))
        if norm > cap and norm > 1e-12:
            return (vec * (cap / norm)).astype(vec.dtype), True
        return vec, False

    def _apply_demo_reset(self, *, fast: bool = False) -> None:
        if self._demo_sampler is None or self._iface is None:
            self._pose_ready = True
            self._last_reset_info = {"demo_reset": False}
            return

        sample = self._demo_sampler.sample_reset_pose()
        target = ResetPose(
            ee_pose=np.asarray(sample["ee_pose"], dtype=np.float32),
            quaternion=np.asarray(sample["quaternion"], dtype=np.float32),
            gripper_width=float(sample["gripper_width"]),
            episode_id=str(sample["episode_id"]),
            success=sample.get("success"),
        )

        safety = self._demo_safety
        if fast and safety is not None:
            safety = replace(safety, require_home_first=False)

        print(
            f"[demo_reset] moving to episode {target.episode_id} "
            f"(demo_index={sample['demo_index']}, pool={len(self._demo_sampler)}, fast={fast})"
        )
        result = move_to_demo_pose(
            self._iface,
            target,
            controller_cfg=self._controller_cfg,
            gripper=self._gripper,
            pos_tol_m=self.cfg.demo_reset_pos_tol_m,
            rot_tol_deg=self.cfg.demo_reset_rot_tol_deg,
            control_hz=self.cfg.control_hz,
            safety=safety,
            joint_controller_cfg=self._joint_controller_cfg,
            osc_position_cfg=self._osc_position_cfg,
        )
        self._pose_ready = True
        self._last_reset_info = {
            "demo_reset": True,
            "demo_reset_fast": fast,
            "demo_index": sample["demo_index"],
            "episode_id": target.episode_id,
            **result,
        }
        print(
            f"[demo_reset] success episode_id={target.episode_id} demo_index={sample['demo_index']}"
        )

    def _apply_workspace_reset(self) -> None:
        if self._iface is None or self._workspace_cfg is None:
            self._pose_ready = True
            self._last_reset_info = {"workspace_reset": False}
            return

        sc = self.cfg.sft_reset_yaml or {}
        print("[collection_reset] init cube reset (same as bash scripts/reset_to_init.sh)")
        ws_result = reset_random_workspace(
            self._iface,
            gripper=self._gripper,
            joint_controller_cfg=self._joint_controller_cfg,
            osc_position_cfg=self._osc_position_cfg,
            ws_cfg=self._workspace_cfg,
            raw=self.cfg.reset_raw or {"sft_collection": sc, "data_collection": self.cfg.demo_reset_yaml},
        )
        self._pose_ready = True
        self._last_reset_info = {
            "workspace_reset": True,
            "episode_id": ws_result.episode_id,
            "target_xyz": ws_result.target_xyz.tolist(),
            "offset_xy_cm": [round(float(x) * 100, 2) for x in ws_result.offset_xy],
            **ws_result.reset_info,
        }
        print(
            f"[collection_reset] target={ws_result.target_xyz.round(4).tolist()} "
            f"offset_xy(cm)={self._last_reset_info['offset_xy_cm']} "
            f"err={ws_result.reset_info.get('pos_err_m', 0)*100:.2f}cm"
        )

    def reset(self, *, fast: bool = False) -> np.ndarray:
        self._step = 0
        self._pose_ready = False
        self._gripper_latched = bool(
            self.cfg.gripper_latch
            and (
                self.cfg.use_workspace_reset
                or self.cfg.demo_reset_gripper_hold_closed
                or not self.cfg.use_demo_reset
            )
        )

        if self.cfg.use_workspace_reset:
            self._apply_workspace_reset()
        elif self.cfg.use_demo_reset:
            self._apply_demo_reset(fast=fast)
        else:
            self._pose_ready = True
            self._last_reset_info = {"demo_reset": False, "workspace_reset": False}

        return self.get_proprio()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        if not self._pose_ready:
            raise RuntimeError("Cannot step before demo reset pose is reached; call reset() first.")

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action, self._gripper_latched = apply_gripper_latch(
            action,
            grasp_pressed=False,
            latched=self._gripper_latched,
            enabled=self.cfg.gripper_latch,
        )
        pos_scale, rot_scale, grip_scale = self.cfg.action_scale
        ctrl_trans, ctrl_rot = self._controller_action_scales()

        safety_info: dict = {}
        if self._iface is None:
            arm_cmd = action[:6].copy()
            if self.cfg.action_is_physical:
                arm_cmd[:3] /= max(ctrl_trans, 1e-8)
                arm_cmd[3:6] /= max(ctrl_rot, 1e-8)
            else:
                arm_cmd[:3] *= pos_scale
                arm_cmd[3:6] *= rot_scale
            arm_cmd, safety_info = self._apply_safety_limits(
                arm_cmd, ctrl_trans=ctrl_trans, ctrl_rot=ctrl_rot
            )
            self._arm.send_cartesian_delta(arm_cmd)
        else:
            cmd = action.copy()
            if self.cfg.action_is_physical:
                # VLA/NPZ deltas in meters — convert to deoxys cmd units (controller applies ctrl_*).
                cmd[:3] /= max(ctrl_trans, 1e-8)
                cmd[3:6] /= max(ctrl_rot, 1e-8)
            else:
                cmd[:3] *= pos_scale
                cmd[3:6] *= rot_scale
            if cmd.shape[0] >= 7 and not self.cfg.action_is_physical:
                cmd[6] *= grip_scale
            cmd, safety_info = self._apply_safety_limits(
                cmd, ctrl_trans=ctrl_trans, ctrl_rot=ctrl_rot
            )
            self._iface.control(
                controller_type=self.cfg.controller_type,
                action=cmd,
                controller_cfg=self._controller_cfg,
            )

        self._step += 1
        info = {
            "step": self._step,
            "gripper_latched": self._gripper_latched,
            **safety_info,
            **self._last_reset_info,
        }
        return self.get_proprio(), 0.0, False, info

    def close(self) -> None:
        self.suspend_deoxys_client()
        if hasattr(self, "_arm") and self.cfg.backend != "deoxys":
            self._arm.close()

    @classmethod
    def from_rlt_config(cls, raw: dict, *, rlt_root: Path | None = None) -> "DeoxysEnv":
        robot = raw.get("robot", {})
        gripper = raw.get("gripper", {})
        rl = raw.get("online_rl", {})
        dc = raw.get("data_collection", {})
        sc = raw.get("sft_collection", {})

        safety = raw.get("safety", {})

        smq = smq_root_from_rlt(rlt_root)
        reset_raw = resolve_reset_yaml(raw, smq_root=smq)
        reset_sc = reset_raw.get("sft_collection", sc)
        reset_dc = reset_raw.get("data_collection", dc)

        reset_mode = rl.get("reset_mode", "")
        use_workspace = bool(
            rl.get("use_workspace_reset", reset_mode in ("workspace", "init_cube", "sft"))
        )
        ws_raw = reset_sc.get("workspace_randomization", dc.get("workspace_randomization", {}))
        workspace_cfg = InitCubeConfig.from_yaml_dict(ws_raw) if ws_raw else None

        demo_path = ""
        if dc.get("use_demo_reset"):
            demo_path = (
                dc.get("demo_reset_path")
                or raw.get("paths", {}).get("demo_reset_path")
                or raw.get("paths", {}).get("episodes_dir", "")
            )
            if demo_path and rlt_root is not None and not Path(demo_path).is_absolute():
                smq = smq_root_from_rlt(rlt_root)
                demo_path = str(resolve_demo_reset_path(raw, smq_root=smq))

        return cls(
            DeoxysEnvConfig(
                backend=robot.get("backend", "dummy"),
                deoxys_root=robot.get("deoxys_root", ""),
                deoxys_config=robot.get("deoxys_config", ""),
                control_hz=robot.get("control_hz", 20.0),
                controller_type=dc.get("controller_type", "OSC_POSE"),
                controller_cfg_name=dc.get("controller_cfg", "osc-position-controller.yml"),
                action_scale=tuple(robot.get("action_scale", [0.05, 0.02, 1.0])),
                proprio_dim=robot.get("proprio_dim", 8),
                action_dim=rl.get("action_dim", 7),
                gripper_config=gripper.get("config", "configs/franka/franka_hand.yaml"),
                chunk_length=rl.get("chunk_length", 10),
                use_demo_reset=bool(dc.get("use_demo_reset", False)) and not use_workspace,
                demo_reset_path=str(demo_path),
                demo_reset_seed=dc.get("demo_reset_seed"),
                demo_reset_pos_tol_m=float(dc.get("demo_reset_pos_tol_m", 0.01)),
                demo_reset_rot_tol_deg=float(dc.get("demo_reset_rot_tol_deg", 5.0)),
                reset_joint_positions=dc.get("reset_joint_positions"),
                demo_reset_yaml=dc,
                joint_controller_cfg_name=dc.get("joint_controller_cfg", "joint-position-controller.yml"),
                gripper_latch=bool(dc.get("gripper_latch", True)),
                demo_reset_gripper_hold_closed=bool(dc.get("demo_reset_gripper_hold_closed", True)),
                demo_reset_pin_episode=dc.get("demo_reset_pin_episode") or None,
                use_workspace_reset=use_workspace,
                workspace_randomization=workspace_cfg,
                sft_reset_yaml={**reset_sc, "fps": reset_sc.get("fps", robot.get("control_hz", 20.0))},
                reset_raw=reset_raw,
                action_is_physical=bool(
                    rl.get("action_is_physical", dc.get("action_is_physical", False))
                ),
                safety_enabled=bool(safety.get("enabled", True)),
                max_trans_delta_m=float(safety.get("max_trans_delta_m", 0.02)),
                max_rot_delta_rad=float(safety.get("max_rot_delta_rad", 0.1)),
                gripper_min=float(safety.get("gripper_min", -1.0)),
                gripper_max=float(safety.get("gripper_max", 1.0)),
            )
        )
