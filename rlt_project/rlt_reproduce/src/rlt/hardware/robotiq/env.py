"""FR3 + Robotiq real-robot environment for RLT critical-phase RL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rlt.hardware.deoxys_arm import ArmBackend, DummyArmBackend, create_deoxys_arm
from rlt.hardware.robotiq.gripper import RobotiqConfig, RobotiqGripperWrapper


@dataclass
class Fr3RobotiqEnvConfig:
    backend: str = "dummy"
    deoxys_root: str = ""
    deoxys_config: str = ""
    control_hz: float = 20.0
    action_scale: tuple[float, float, float] = (0.05, 0.02, 1.0)
    proprio_dim: int = 8
    action_dim: int = 7
    gripper_config: str = "configs/robotiq/robotiq.yaml"
    chunk_length: int = 10


class Fr3RobotiqEnv:
    """Minimal env: deoxys arm + Robotiq gripper."""

    def __init__(self, cfg: Fr3RobotiqEnvConfig):
        self.cfg = cfg
        self._arm: ArmBackend
        if cfg.backend == "deoxys":
            self._arm = create_deoxys_arm(
                cfg.deoxys_root,
                cfg.deoxys_config,
                control_hz=cfg.control_hz,
                has_gripper=False,
            )
        else:
            self._arm = DummyArmBackend()
        gripper_cfg = RobotiqConfig.from_yaml(Path(cfg.gripper_config))
        self._gripper = RobotiqGripperWrapper(gripper_cfg)
        self._step = 0

    def get_proprio(self) -> np.ndarray:
        arm = self._arm.get_state()
        width = np.array([self._gripper.position], dtype=np.float32)
        return np.concatenate([arm.ee_pose, width]).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        pos_scale, rot_scale, grip_scale = self.cfg.action_scale
        arm_cmd = action[:6].copy()
        arm_cmd[:3] *= pos_scale
        arm_cmd[3:6] *= rot_scale
        self._arm.send_cartesian_delta(arm_cmd)
        if action.shape[0] >= 7:
            self._gripper.apply_action(float(action[6]) * grip_scale)
        self._step += 1
        proprio = self.get_proprio()
        return proprio, 0.0, False, {"step": self._step}

    def reset(self) -> np.ndarray:
        self._step = 0
        return self.get_proprio()

    def close(self) -> None:
        self._gripper.cleanup()
        self._arm.close()

    @classmethod
    def from_rlt_config(cls, raw: dict) -> "Fr3RobotiqEnv":
        robot = raw.get("robot", {})
        gripper = raw.get("gripper", {})
        rl = raw.get("online_rl", {})
        return cls(
            Fr3RobotiqEnvConfig(
                backend=robot.get("backend", "dummy"),
                deoxys_root=robot.get("deoxys_root", ""),
                deoxys_config=robot.get("deoxys_config", ""),
                control_hz=robot.get("control_hz", 20.0),
                action_scale=tuple(robot.get("action_scale", [0.05, 0.02, 1.0])),
                proprio_dim=robot.get("proprio_dim", 8),
                action_dim=rl.get("action_dim", 7),
                gripper_config=gripper.get("config", "configs/robotiq/robotiq.yaml"),
                chunk_length=rl.get("chunk_length", 10),
            )
        )
