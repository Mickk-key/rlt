"""Deoxys FR3 environment with optional demo-driven reset."""

from __future__ import annotations

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
from rlt.hardware.deoxys_arm import DummyArmBackend, o_t_ee_to_pose
from rlt.hardware.franka.gripper import FrankaConfig, FrankaGripperAdapter
from rlt.teleop.spacemouse_control import apply_gripper_latch
from rlt.util.deoxys_paths import (
    default_osc_controller_cfg_name,
    resolve_controller_cfg_path,
    smq_root_from_rlt,
)


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

            self._iface = FrankaInterface(
                interface_path,
                control_freq=cfg.control_hz,
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
        else:
            self._arm = DummyArmBackend()

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

    def reset(self, *, fast: bool = False) -> np.ndarray:
        self._step = 0
        self._pose_ready = False
        self._gripper_latched = bool(
            self.cfg.gripper_latch
            and (self.cfg.demo_reset_gripper_hold_closed or not self.cfg.use_demo_reset)
        )

        if self.cfg.use_demo_reset:
            self._apply_demo_reset(fast=fast)
        else:
            self._pose_ready = True
            self._last_reset_info = {"demo_reset": False}

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

        if self._iface is None:
            arm_cmd = action[:6].copy()
            arm_cmd[:3] *= pos_scale
            arm_cmd[3:6] *= rot_scale
            self._arm.send_cartesian_delta(arm_cmd)
        else:
            cmd = action.copy()
            cmd[:3] *= pos_scale
            cmd[3:6] *= rot_scale
            if cmd.shape[0] >= 7:
                cmd[6] *= grip_scale
            self._iface.control(
                controller_type=self.cfg.controller_type,
                action=cmd,
                controller_cfg=self._controller_cfg,
            )

        self._step += 1
        info = {
            "step": self._step,
            "gripper_latched": self._gripper_latched,
            **self._last_reset_info,
        }
        return self.get_proprio(), 0.0, False, info

    def close(self) -> None:
        if self._gripper is not None:
            self._gripper.cleanup()
        if self._iface is not None:
            self._iface.close()
        elif hasattr(self, "_arm"):
            self._arm.close()

    @classmethod
    def from_rlt_config(cls, raw: dict, *, rlt_root: Path | None = None) -> "DeoxysEnv":
        robot = raw.get("robot", {})
        gripper = raw.get("gripper", {})
        rl = raw.get("online_rl", {})
        dc = raw.get("data_collection", {})

        demo_path = dc.get("demo_reset_path") or raw.get("paths", {}).get("demo_reset_path") or raw.get("paths", {}).get("episodes_dir", "")
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
                use_demo_reset=bool(dc.get("use_demo_reset", False)),
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
            )
        )
