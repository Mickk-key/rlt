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
        success_lift_z_m: float = 0.0,
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
        self.success_lift_z_m = float(success_lift_z_m)
        self.last_lift_info: dict = {"applied": False}

    def _lift_after_success(self) -> dict:
        """Raise EE straight up before归位 when the previous episode succeeded.

        Runs on the actor's own Deoxys client (before any external reset
        subprocess suspends it), so the still-gripped plug is pulled vertically
        out of the socket instead of being dragged sideways by xy-first reset.

        Returns a dict with before/after EE positions (for logging / tests).
        """
        info: dict = {"applied": False, "lift_z_m": self.success_lift_z_m}
        if self.success_lift_z_m <= 0.0:
            print(f"[reset] success z-lift disabled (success_lift_z_m={self.success_lift_z_m})")
            return info
        iface = self.env._iface
        if iface is None or self.env._osc_position_cfg is None:
            print("[reset] success z-lift skipped — no live Deoxys client (iface/osc cfg is None)")
            return info

        from rlt.hardware.deoxys.collection_reset import collection_reset_settings
        from rlt.hardware.deoxys.fast_reset import lift_ee_z

        _, fast_cfg = collection_reset_settings(self.reset_raw)
        before = self.env.get_proprio()[:3].astype(float)
        print(
            f"[reset] SUCCESS-LIFT branch entered: raising EE +{self.success_lift_z_m*100:.1f}cm "
            f"in z (xy locked, gripper closed) from ee_xyz={before.round(4).tolist()}"
        )
        try:
            steps, pos_err = lift_ee_z(
                iface,
                lift_m=self.success_lift_z_m,
                osc_position_cfg=self.env._osc_position_cfg,
                reset_cfg=fast_cfg,
            )
            after = self.env.get_proprio()[:3].astype(float)
            info.update(
                applied=True,
                steps=int(steps),
                pos_err_m=float(pos_err),
                before_xyz=before.tolist(),
                after_xyz=after.tolist(),
                delta_xyz=(after - before).tolist(),
            )
            print(
                f"[reset] success z-lift done steps={steps} pos_err={pos_err*100:.2f}cm "
                f"ee_xyz={after.round(4).tolist()} "
                f"Δxyz(cm)={((after - before) * 100).round(2).tolist()}"
            )
        except Exception as exc:  # never block reset on a lift failure
            info["error"] = str(exc)
            print(f"[reset] success z-lift skipped ({exc})")
        return info

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

    def reset(self, *, prev_success: bool = False) -> tuple[np.ndarray, dict]:
        """Run reset pipeline; return initial proprio and metadata.

        When ``prev_success`` is set, first raise the EE straight up (z only) so
        the gripped plug leaves the socket vertically before any归位 motion.
        """
        import time

        info: dict = {"reset_mode": self.mode.value}

        print(f"[reset] prev_success={prev_success} success_lift_z_m={self.success_lift_z_m}")
        self.last_lift_info: dict = {"applied": False}
        if prev_success:
            self.last_lift_info = self._lift_after_success()
            info["success_lift"] = self.last_lift_info
        else:
            print("[reset] prev_success=False → skipping success z-lift, going straight to归位")

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
            success_lift_z_m=float(online.get("success_lift_z_m", 0.06)),
        )
