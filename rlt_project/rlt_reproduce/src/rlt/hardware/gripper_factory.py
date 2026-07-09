"""Select Robotiq vs Franka Hand stack from YAML config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rlt.hardware.franka.gripper import FrankaConfig, FrankaGripperAdapter
from rlt.hardware.robotiq.gripper import RobotiqConfig, RobotiqGripperWrapper


def gripper_type(raw: dict) -> str:
    return str(raw.get("gripper", {}).get("type", "franka")).lower()


def uses_deoxys_gripper(raw: dict) -> bool:
    return gripper_type(raw) == "franka"


def create_gripper(raw: dict, robot_interface: Any | None = None):
    gtype = gripper_type(raw)
    cfg_path = raw.get("gripper", {}).get("config")
    if gtype == "robotiq":
        path = cfg_path or "configs/robotiq/robotiq.yaml"
        return RobotiqGripperWrapper(RobotiqConfig.from_yaml(Path(path)))
    if robot_interface is None:
        raise ValueError("Franka Hand requires an active deoxys FrankaInterface")
    path = cfg_path or "configs/franka/franka_hand.yaml"
    return FrankaGripperAdapter(robot_interface, FrankaConfig.from_yaml(Path(path)))


def create_robot_env(raw: dict, *, rlt_root: Path | None = None):
    """Build FR3 env for the configured gripper backend."""
    if gripper_type(raw) == "robotiq":
        from rlt.hardware.robotiq.env import Fr3RobotiqEnv

        return Fr3RobotiqEnv.from_rlt_config(raw)
    from rlt.hardware.deoxys.deoxys_env import DeoxysEnv

    if rlt_root is None:
        episodes = raw.get("paths", {}).get("episodes_dir")
        if episodes:
            rlt_root = Path(episodes).resolve().parent.parent
        else:
            rlt_root = Path.cwd()
    return DeoxysEnv.from_rlt_config(raw, rlt_root=rlt_root)
