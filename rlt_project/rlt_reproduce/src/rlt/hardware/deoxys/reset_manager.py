"""Episode reset orchestration for online RL (not part of the RL step)."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rlt.hardware.deoxys.collection_reset import (
    resolve_reset_yaml,
    run_external_reset_subprocess,
    wait_until_init_pose,
)
from rlt.teleop.spacemouse_control import move_arm_to_reset_pose
from rlt.util.deoxys_paths import smq_root_from_rlt

if TYPE_CHECKING:
    from rlt.hardware.deoxys.deoxys_env import DeoxysEnv


class ResetMode(str, Enum):
    DEMO = "demo"
    DEMO_FAST = "demo_fast"
    WORKSPACE = "workspace"
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
        reset_method: str = "external",
        reset_config_path: Path | None = None,
        smq_root: Path | None = None,
        rlt_root: Path | None = None,
        reset_raw: dict | None = None,
    ) -> None:
        self.env = env
        self.mode = ResetMode(mode)
        self.home_joints = home_joints
        self.post_reset_wait_sec = post_reset_wait_sec
        self.reset_method = str(reset_method).lower()
        self.reset_config_path = reset_config_path
        self.smq_root = smq_root
        self.rlt_root = rlt_root
        self.reset_raw = reset_raw or {}

    def _prepare_episode_gripper(self) -> None:
        """Latch gripper closed for critical-phase rollout when configured."""
        if self.env.cfg.gripper_latch:
            self.env._gripper_latched = True

    def _external_workspace_reset(self) -> tuple[np.ndarray, dict]:
        import time

        if self.smq_root is None or self.rlt_root is None or self.reset_config_path is None:
            raise RuntimeError("external reset requires smq_root, rlt_root, reset_config_path")

        print("[external_reset] closing actor Deoxys client — same flow as reset_to_init.sh")
        self.env.suspend_deoxys_client()

        run_external_reset_subprocess(
            smq_root=self.smq_root,
            rlt_root=self.rlt_root,
            config_path=self.reset_config_path,
            randomize=True,
        )

        print("[external_reset] reconnecting actor Deoxys client for RL rollout")
        self.env.resume_deoxys_client()
        self.env._pose_ready = True
        self._prepare_episode_gripper()

        if self.post_reset_wait_sec > 0:
            time.sleep(self.post_reset_wait_sec)

        proprio = wait_until_init_pose(
            self.env.get_proprio,
            self.reset_raw,
            timeout_sec=30.0,
        )
        pos = proprio[:3]
        info = {
            "reset_mode": "workspace",
            "workspace_reset": True,
            "external_reset": True,
            "target_xyz": pos.tolist(),
            "live_pos_err_m": 0.0,
            "pos_err_m": 0.0,
            "ee_z_m": float(pos[2]),
        }
        print(
            f"[external_reset] verified init pose ee_xyz={np.round(pos, 4).tolist()} "
            f"z={float(pos[2]):.4f}m — OK to start VLA"
        )
        self.env._last_reset_info = dict(info)
        return proprio, info

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
            self._prepare_episode_gripper()
            return proprio, info

        if self.mode == ResetMode.NONE:
            proprio = self.env.get_proprio()
            info["skipped"] = True
            self._prepare_episode_gripper()
            return proprio, info

        if self.mode == ResetMode.WORKSPACE:
            if self.reset_method == "external":
                return self._external_workspace_reset()

            proprio = self.env.reset()
            info.update(self.env.last_reset_info)
            info["reset_mode"] = "workspace"
            if self.post_reset_wait_sec > 0:
                time.sleep(self.post_reset_wait_sec)
            if self.env._iface is not None:
                proprio = wait_until_init_pose(self.env.get_proprio, self.reset_raw, timeout_sec=20.0)
                info["target_xyz"] = proprio[:3].tolist()
                info["ee_z_m"] = float(proprio[2])
            self._prepare_episode_gripper()
            return proprio, info

        if self.mode == ResetMode.DEMO_FAST:
            proprio = self.env.reset(fast=True)
            info.update(self.env.last_reset_info)
            info["reset_mode"] = "demo_fast"
            self._prepare_episode_gripper()
            return proprio, info

        proprio = self.env.reset(fast=False)
        info.update(self.env.last_reset_info)
        self._prepare_episode_gripper()
        return proprio, info

    @classmethod
    def from_config(cls, env: DeoxysEnv, raw: dict, *, rlt_root: Path | None = None) -> ResetManager:
        dc = raw.get("data_collection", {})
        online = raw.get("online_rl", {})
        sc = raw.get("sft_collection", {})
        smq = smq_root_from_rlt(rlt_root)
        reset_raw = resolve_reset_yaml(raw, smq_root=smq)

        mode = online.get("reset_mode")
        if mode is None:
            if dc.get("use_demo_reset"):
                mode = ResetMode.DEMO.value
            elif online.get("use_workspace_reset") or sc.get("workspace_randomization"):
                mode = ResetMode.WORKSPACE.value
            else:
                mode = ResetMode.HOME.value

        reset_config_rel = online.get("reset_config", "configs/sft_plug_insertion.yaml")
        reset_config_path = Path(reset_config_rel)
        if not reset_config_path.is_absolute():
            reset_config_path = (smq / reset_config_rel).resolve()

        return cls(
            env,
            mode=mode,
            home_joints=dc.get("reset_joint_positions"),
            post_reset_wait_sec=float(online.get("post_reset_wait_sec", 0.5)),
            reset_method=str(online.get("reset_method", "external")),
            reset_config_path=reset_config_path,
            smq_root=smq,
            rlt_root=rlt_root,
            reset_raw=reset_raw,
        )
