"""Robotiq gripper wrapper for RLT + deoxys arm setups."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rlt.hardware.robotiq.modbus import RobotiqGripper


@dataclass
class RobotiqConfig:
    gripper_type: str = "robotiq"
    serial_port: str = "/dev/ttyUSB0"
    serial_port_by_id: str | None = None
    baudrate: int = 115200
    slave_id: int = 9
    max_width: float = 0.085
    rlinf_root: str = "/home/host5010/workspaces/yjr/ConsVLA/RLinf"
    binary_open_threshold: float = 0.5
    binary_close_threshold: float = -0.5

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RobotiqConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def resolve_port(self) -> str:
        if self.serial_port_by_id and Path(self.serial_port_by_id).exists():
            return self.serial_port_by_id
        return self.serial_port


class RobotiqGripperWrapper:
    """Robotiq driver facade for RLT real-robot integration."""

    def __init__(self, config: RobotiqConfig | None = None, **kwargs: Any):
        if config is None:
            config = RobotiqConfig(**kwargs)
        self.config = config
        port = config.resolve_port()
        self._gripper = RobotiqGripper(
            port=port,
            baudrate=config.baudrate,
            slave_id=config.slave_id,
            max_width=config.max_width,
        )
        if not self._gripper.is_ready():
            raise RuntimeError(f"Robotiq activation failed on {port}")

    @property
    def position(self) -> float:
        return float(self._gripper.position)

    @property
    def is_open(self) -> bool:
        return bool(self._gripper.is_open)

    def open(self) -> None:
        self._gripper.open()

    def close(self) -> None:
        self._gripper.close()

    def apply_action(self, gripper_action: float) -> bool:
        """Binary open/close from a scalar action (Franka2Env convention)."""
        if gripper_action <= self.config.binary_close_threshold and self.is_open:
            self.close()
            return True
        if gripper_action >= self.config.binary_open_threshold and not self.is_open:
            self.open()
            return True
        return False

    def cleanup(self) -> None:
        self._gripper.cleanup()

    @classmethod
    def from_default_config(cls) -> "RobotiqGripperWrapper":
        default = Path(__file__).resolve().parents[4] / "configs" / "robotiq" / "robotiq.yaml"
        env_path = os.environ.get("RLT_ROBOTIQ_CONFIG")
        path = Path(env_path) if env_path else default
        return cls(RobotiqConfig.from_yaml(path))
