"""Backward-compatible re-exports — use ``rlt.hardware.robotiq`` instead."""

from rlt.hardware.robotiq.gripper import RobotiqConfig, RobotiqGripperWrapper

__all__ = ["RobotiqConfig", "RobotiqGripperWrapper"]
