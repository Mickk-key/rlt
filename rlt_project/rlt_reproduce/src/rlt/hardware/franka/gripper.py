"""Franka Hand gripper via deoxys FrankaInterface (ZMQ to gripper-interface)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class FrankaConfig:
    gripper_type: str = "franka"
    max_width: float = 0.08
    binary_open_threshold: float = 0.5
    binary_close_threshold: float = -0.5

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FrankaConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class FrankaGripperAdapter:
    """Read Franka Hand state from an active deoxys FrankaInterface.

    Gripper commands are sent through ``FrankaInterface.control(..., action[-1])``
    when ``has_gripper=True``; this adapter only exposes width for proprioception.
    """

    def __init__(self, robot_interface: Any, config: FrankaConfig | None = None):
        self._robot = robot_interface
        self.config = config or FrankaConfig()

    @property
    def position(self) -> float:
        width = self._robot.last_gripper_q
        if width is None:
            return self.config.max_width
        return float(np.asarray(width).reshape(-1)[0])

    @property
    def is_open(self) -> bool:
        return self.position > self.config.max_width * 0.5

    def apply_action(self, gripper_action: float) -> bool:
        """No-op: deoxys controls the Franka Hand inside ``FrankaInterface.control``."""
        return False

    def cleanup(self) -> None:
        pass
