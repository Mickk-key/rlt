"""Episode reset orchestration for online RL (not part of the RL step)."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from rlt.teleop.spacemouse_control import move_arm_to_reset_pose

if TYPE_CHECKING:
    from rlt.hardware.deoxys.deoxys_env import DeoxysEnv


class ResetMode(str, Enum):
    DEMO = "demo"
    HOME = "home"
    NONE = "none"


class ResetManager:
    """Select reset strategy before each online RL episode.

    Reset belongs to the environment layer, not the learner.
    """

    def __init__(
        self,
        env: DeoxysEnv,
        *,
        mode: ResetMode | str = ResetMode.DEMO,
        home_joints: list[float] | None = None,
        post_reset_wait_sec: float = 0.5,
    ) -> None:
        self.env = env
        self.mode = ResetMode(mode)
        self.home_joints = home_joints
        self.post_reset_wait_sec = post_reset_wait_sec

    def reset(self) -> tuple[np.ndarray, dict]:
        """Run reset pipeline; return initial proprio and metadata."""
        import time

        info: dict = {"reset_mode": self.mode.value}

        if self.mode == ResetMode.HOME:
            if self.env._iface is None:
                proprio = self.env.reset()
                info["home_reset"] = False
                info.update(self.env.last_reset_info)
                return proprio, info

            move_arm_to_reset_pose(
                self.env._iface,
                self.home_joints or self.env.cfg.reset_joint_positions,
                controller_cfg=self.env._joint_controller_cfg,
            )
            if self.post_reset_wait_sec > 0:
                time.sleep(self.post_reset_wait_sec)
            proprio = self.env.get_proprio()
            info["home_reset"] = True
            info.update(self.env.last_reset_info)
            return proprio, info

        if self.mode == ResetMode.NONE:
            proprio = self.env.get_proprio()
            info["skipped"] = True
            return proprio, info

        # DEMO: DeoxysEnv.reset() runs DemoResetSampler + move_to_demo_pose when enabled.
        proprio = self.env.reset()
        info.update(self.env.last_reset_info)
        return proprio, info

    @classmethod
    def from_config(cls, env: DeoxysEnv, raw: dict) -> ResetManager:
        dc = raw.get("data_collection", {})
        online = raw.get("online_rl", {})
        mode = online.get("reset_mode")
        if mode is None:
            mode = ResetMode.DEMO.value if dc.get("use_demo_reset") else ResetMode.HOME.value
        return cls(
            env,
            mode=mode,
            home_joints=dc.get("reset_joint_positions"),
            post_reset_wait_sec=float(online.get("post_reset_wait_sec", 0.5)),
        )
