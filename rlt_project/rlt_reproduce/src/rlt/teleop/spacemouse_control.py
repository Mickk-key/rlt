"""SpaceMouse teleop helpers for FR3 + Franka Hand (smq workspace wrappers)."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

# Initial joint pose from docs/desktop/deoxys.txt (拍照/采集起始位姿)
DEFAULT_RESET_JOINTS: list[float] = [
    0.05904578,
    0.52446592,
    -0.15976057,
    -0.76166065,
    0.18787272,
    1.7980539,
    0.81746185,
]


def move_arm_to_reset_pose(
    robot_interface,
    reset_joints: Sequence[float] | None = None,
    *,
    controller_cfg=None,
    gripper_open: bool = False,
    timeout: float = 12.0,
) -> None:
    """Move arm to a fixed joint configuration without closing the teleop session."""
    from deoxys.experimental.motion_utils import reset_joints_to
    from deoxys.utils.config_utils import get_default_controller_config

    joints = list(reset_joints or DEFAULT_RESET_JOINTS)
    cfg = controller_cfg
    if cfg is None:
        cfg = get_default_controller_config(controller_type="JOINT_POSITION")
    reset_joints_to(
        robot_interface,
        joints,
        controller_cfg=cfg,
        timeout=timeout,
        gripper_open=gripper_open,
    )


def open_franka_gripper(robot_interface, *, hold_sec: float = 0.8) -> None:
    """Send open command; useful on program exit."""
    if not getattr(robot_interface, "has_gripper", False):
        return
    try:
        robot_interface.gripper_control(-1.0)
        if hold_sec > 0:
            time.sleep(hold_sec)
    except Exception:
        pass


def apply_gripper_latch(
    action: np.ndarray,
    *,
    grasp_pressed: bool,
    latched: bool,
    enabled: bool,
) -> tuple[np.ndarray, bool]:
    """Once latched, keep sending close command until explicitly opened."""
    if not enabled:
        return action, latched
    if grasp_pressed:
        latched = True
    if latched:
        action = action.copy()
        action[-1] = 1.0
    return action, latched


def is_spacemouse_reset(action) -> bool:
    return action is None


def acknowledge_spacemouse_reset(device) -> None:
    """After handling SpaceMouse RIGHT in our code, re-enable teleop.

    deoxys sets ``_reset_state=1`` and ``_enabled=False`` on button 2; if we do not
    clear these, every subsequent ``input2action`` returns ``None`` and motion stops.
    """
    device._reset_state = 0
    device._enabled = True
