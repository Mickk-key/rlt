"""Optional deoxys arm backend for FR3 (gripper via Franka Hand or external Robotiq)."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class ArmState:
    ee_pose: np.ndarray  # (7,) xyz + quat(w,x,y,z)
    joint_positions: np.ndarray  # (7,)


class ArmBackend(Protocol):
    def get_state(self) -> ArmState: ...
    def send_cartesian_delta(self, action6: np.ndarray) -> None: ...
    def close(self) -> None: ...


class DummyArmBackend:
    """Safe stand-in when robot is unavailable."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self._ee = np.array([0.45, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._q = np.zeros(7, dtype=np.float32)

    def get_state(self) -> ArmState:
        return ArmState(ee_pose=self._ee.copy(), joint_positions=self._q.copy())

    def send_cartesian_delta(self, action6: np.ndarray) -> None:
        self._ee[:3] += action6[:3]
        self._ee[3:] += action6[3:] * 0.01

    def close(self) -> None:
        pass


def create_deoxys_arm(
    deoxys_root: str,
    config_path: str,
    control_hz: float = 20.0,
    has_gripper: bool = False,
) -> ArmBackend:
    root = Path(deoxys_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from deoxys.franka_interface.franka_interface import FrankaInterface

    iface = FrankaInterface(
        general_cfg_file=config_path,
        control_freq=control_hz,
        has_gripper=has_gripper,
        automatic_gripper_reset=False,
    )
    return DeoxysArmBackend(iface)


class DeoxysArmBackend:
    def __init__(self, iface):
        self._iface = iface
        time.sleep(0.5)

    def _latest_state(self):
        if not self._iface._state_buffer:
            raise RuntimeError("No deoxys state yet — is auto_arm.sh running?")
        return self._iface._state_buffer[-1]

    def get_state(self) -> ArmState:
        st = self._latest_state()
        pos, quat = o_t_ee_to_pose(st.O_T_EE)
        q = np.array(st.q, dtype=np.float32)
        return ArmState(ee_pose=np.concatenate([pos, quat]), joint_positions=q)

    def send_cartesian_delta(self, action6: np.ndarray) -> None:
        goal = np.asarray(action6[:6], dtype=np.float64)
        self._iface.control("OSC_POSE", goal)

    def close(self) -> None:
        self._iface.termination = True


def o_t_ee_to_pose(o_t_ee_flat) -> tuple[np.ndarray, np.ndarray]:
    """Parse libfranka O_T_EE (column-major flat 16) into position and quaternion."""
    o_t_ee = np.array(o_t_ee_flat, dtype=np.float32).reshape(4, 4).transpose()
    pos = o_t_ee[:3, 3]
    quat = _rotmat_to_quat(o_t_ee[:3, :3])
    return pos, quat


def _rotmat_to_quat(rot: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    m = rot
    trace = float(np.trace(m))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float32)
    return q / (np.linalg.norm(q) + 1e-8)
