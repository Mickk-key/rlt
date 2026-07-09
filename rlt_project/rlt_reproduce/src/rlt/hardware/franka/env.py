"""FR3 + Franka Hand environment for RLT critical-phase RL."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rlt.hardware.deoxys_arm import DummyArmBackend, _rotmat_to_quat
from rlt.hardware.franka.gripper import FrankaConfig, FrankaGripperAdapter


@dataclass
class Fr3FrankaEnvConfig:
    backend: str = "dummy"
    deoxys_root: str = ""
    deoxys_config: str = ""
    control_hz: float = 20.0
    controller_type: str = "OSC_POSE"
    action_scale: tuple[float, float, float] = (0.05, 0.02, 1.0)
    proprio_dim: int = 8
    action_dim: int = 7
    gripper_config: str = "configs/franka/franka_hand.yaml"
    chunk_length: int = 10


class Fr3FrankaEnv:
    """deoxys arm + native Franka Hand (gripper via FrankaInterface)."""

    def __init__(self, cfg: Fr3FrankaEnvConfig):
        self.cfg = cfg
        self._iface = None
        self._gripper: FrankaGripperAdapter | None = None
        self._step = 0

        if cfg.backend == "deoxys":
            root = Path(cfg.deoxys_root).resolve()
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from deoxys.franka_interface import FrankaInterface
            from deoxys.utils.config_utils import get_default_controller_config

            self._iface = FrankaInterface(
                cfg.deoxys_config,
                control_freq=cfg.control_hz,
                has_gripper=True,
                automatic_gripper_reset=False,
            )
            self._controller_cfg = get_default_controller_config(cfg.controller_type)
            gripper_cfg = FrankaConfig.from_yaml(Path(cfg.gripper_config))
            self._gripper = FrankaGripperAdapter(self._iface, gripper_cfg)
            time.sleep(0.5)
        else:
            self._arm = DummyArmBackend()

    def get_proprio(self) -> np.ndarray:
        if self._iface is None:
            arm = self._arm.get_state()
            width = np.array([0.04], dtype=np.float32)
            return np.concatenate([arm.ee_pose, width]).astype(np.float32)

        st = self._iface._state_buffer[-1]
        o_t_ee = np.array(st.O_T_EE, dtype=np.float32).reshape(4, 4)
        pos = o_t_ee[:3, 3]
        quat = _rotmat_to_quat(o_t_ee[:3, :3])
        width = np.array([self._gripper.position], dtype=np.float32)
        return np.concatenate([pos, quat, width]).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
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
        return self.get_proprio(), 0.0, False, {"step": self._step}

    def reset(self) -> np.ndarray:
        self._step = 0
        return self.get_proprio()

    def close(self) -> None:
        if self._gripper is not None:
            self._gripper.cleanup()
        if self._iface is not None:
            self._iface.close()
        elif hasattr(self, "_arm"):
            self._arm.close()

    @classmethod
    def from_rlt_config(cls, raw: dict) -> "Fr3FrankaEnv":
        robot = raw.get("robot", {})
        gripper = raw.get("gripper", {})
        rl = raw.get("online_rl", {})
        dc = raw.get("data_collection", {})
        return cls(
            Fr3FrankaEnvConfig(
                backend=robot.get("backend", "dummy"),
                deoxys_root=robot.get("deoxys_root", ""),
                deoxys_config=robot.get("deoxys_config", ""),
                control_hz=robot.get("control_hz", 20.0),
                controller_type=dc.get("controller_type", "OSC_POSE"),
                action_scale=tuple(robot.get("action_scale", [0.05, 0.02, 1.0])),
                proprio_dim=robot.get("proprio_dim", 8),
                action_dim=rl.get("action_dim", 7),
                gripper_config=gripper.get("config", "configs/franka/franka_hand.yaml"),
                chunk_length=rl.get("chunk_length", 10),
            )
        )
