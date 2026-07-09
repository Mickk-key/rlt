#!/usr/bin/env python3
"""Collect plug-insertion critical-phase demos: deoxys arm + gripper + RealSense.

Designed for RLT reproduction without GPU on the robot PC.
Save NPZ episodes here; copy `data/episodes/plug_insertion/` to the GPU host for training.

Controls (OpenCV window `rlt_collect` must be focused for keys):
  r  start recording (enter critical phase — aligned, about to insert)
  s  save episode as SUCCESS
  f  save episode as FAILURE
  q  quit

SpaceMouse: translate/rotate arm; left/right buttons open/close gripper.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from rich.console import Console

from rlt.data.episode_io import save_episode
from rlt.data.schema import CriticalPhaseEpisode, EpisodeMetadata
from rlt.hardware.gripper_factory import create_gripper, uses_deoxys_gripper
from rlt.teleop.spacemouse_control import (
    DEFAULT_RESET_JOINTS,
    acknowledge_spacemouse_reset,
    apply_gripper_latch,
    is_spacemouse_reset,
    move_arm_to_reset_pose,
    open_franka_gripper,
)

console = Console()


def _setup_deoxys(deoxys_root: str):
    root = Path(deoxys_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from deoxys import config_root
    from deoxys.camera import RealSenseCameraManager
    from deoxys.franka_interface import FrankaInterface
    from deoxys.utils import YamlConfig
    from deoxys.utils.config_utils import get_default_controller_config
    from deoxys.utils.input_utils import input2action
    from deoxys.utils.io_devices import SpaceMouse

    return (
        config_root,
        RealSenseCameraManager,
        FrankaInterface,
        YamlConfig,
        get_default_controller_config,
        input2action,
        SpaceMouse,
    )


def _rotmat_to_quat(rot: np.ndarray) -> np.ndarray:
    m = rot
    trace = float(np.trace(m))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w, x = 0.25 / s, (m[2, 1] - m[1, 2]) * s
        y, z = (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w, x = (m[2, 1] - m[1, 2]) / s, 0.25 * s
        y, z = (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w, x = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s
        y, z = 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w, x = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s
        y, z = (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float32)
    return q / (np.linalg.norm(q) + 1e-8)


def _proprio_from_state(robot_interface, gripper_width: float) -> np.ndarray:
    st = robot_interface._state_buffer[-1]
    o_t_ee = np.array(st.O_T_EE, dtype=np.float32).reshape(4, 4)
    pos = o_t_ee[:3, 3]
    quat = _rotmat_to_quat(o_t_ee[:3, :3])
    return np.concatenate([pos, quat, [gripper_width]]).astype(np.float32)


def _normalize_action(action7: np.ndarray, pos_scale: float, rot_scale: float) -> np.ndarray:
    out = action7.copy()
    out[:3] *= pos_scale
    out[3:6] *= rot_scale
    return out.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/plug_insertion.yaml"))
    parser.add_argument(
        "--interface-cfg",
        type=str,
        default=None,
        help="Override deoxys yaml (default from config robot.deoxys_config)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    robot_cfg = raw["robot"]
    dc = raw["data_collection"]
    cam_cfg = raw["cameras"]
    paths = raw["paths"]
    language = raw["vla"]["language_instruction"]
    gripper_type = raw.get("gripper", {}).get("type", "franka")

    fps = float(dc.get("fps", 20))
    dt = 1.0 / fps
    episodes_dir = Path(paths["episodes_dir"])
    episodes_dir.mkdir(parents=True, exist_ok=True)

    deoxys_root = robot_cfg["deoxys_root"]
    interface_cfg = args.interface_cfg or robot_cfg["deoxys_config"]
    controller_type = dc.get("controller_type", "OSC_POSE")
    controller_cfg_name = dc.get("controller_cfg", "osc-position-controller.yml")
    pos_scale, rot_scale, _ = robot_cfg.get("action_scale", [0.05, 0.02, 1.0])

    (
        config_root,
        RealSenseCameraManager,
        FrankaInterface,
        YamlConfig,
        get_default_controller_config,
        input2action,
        SpaceMouse,
    ) = _setup_deoxys(deoxys_root)

    if not interface_cfg.startswith("/"):
        interface_path = os.path.join(config_root, Path(interface_cfg).name)
        if not os.path.isfile(interface_path):
            interface_path = interface_cfg
    else:
        interface_path = interface_cfg

    has_deoxys_gripper = uses_deoxys_gripper(raw)
    robot = FrankaInterface(
        interface_path,
        control_freq=fps,
        has_gripper=has_deoxys_gripper,
        automatic_gripper_reset=False,
    )
    gripper = create_gripper(raw, robot_interface=robot)

    controller_cfg = get_default_controller_config(controller_type)
    if controller_cfg_name:
        user_cfg = YamlConfig(os.path.join(config_root, controller_cfg_name)).as_easydict()
        controller_cfg = user_cfg
    joint_reset_cfg = YamlConfig(os.path.join(config_root, "joint-position-controller.yml")).as_easydict()

    device = SpaceMouse()
    device.start_control()

    camera_mapping = cam_cfg.get("mapping", {})
    camera_manager = None
    if camera_mapping:
        camera_manager = RealSenseCameraManager(
            camera_name_mapping=camera_mapping,
            camera_width=int(cam_cfg.get("width", 640)),
            camera_height=int(cam_cfg.get("height", 480)),
            camera_fps=int(fps),
        )
        console.print(f"Cameras: {camera_mapping}")

    img_size = tuple(cam_cfg.get("image_size", [224, 224]))
    console.print(f"[bold]Plug insertion collection[/bold] -> {episodes_dir}")
    console.print(f"Gripper backend: {gripper_type}")
    console.print("SpaceMouse move arm | L/R = open/close gripper")
    console.print("[cyan]r[/cyan]=start record  [green]s[/green]=success save  [red]f[/red]=fail save  [yellow]q[/yellow]=quit")
    console.print("Record ONLY the critical insertion phase (last ~1-3 s before seated).")

    cv2.namedWindow("rlt_collect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("rlt_collect", 320, 120)
    placeholder = np.zeros((120, 320, 3), dtype=np.uint8)
    cv2.putText(placeholder, "focus here for keys", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    recording = False
    buffers: dict[str, list] = {
        k: [] for k in ["timestamps", "proprio", "actions", "rewards", "dones", "is_human", "wrist", "external"]
    }
    t0 = 0.0
    episode_count = 0

    try:
        while True:
            action_raw, grasp = input2action(device=device, controller_type=controller_type)
            if action_raw is None:
                break
            action_raw = np.asarray(action_raw, dtype=np.float32)
            if controller_type == "OSC_POSITION":
                action_raw[3:6] = 0.0

            robot.control(controller_type=controller_type, action=action_raw, controller_cfg=controller_cfg)
            if not has_deoxys_gripper:
                gripper.apply_action(float(action_raw[-1]))

            key = cv2.waitKey(1) & 0xFF
            cv2.imshow("rlt_collect", placeholder)
            if key == ord("q"):
                break
            if key == ord("r") and not recording:
                recording = True
                buffers = {k: [] for k in buffers}
                t0 = time.time()
                console.print("[cyan]Recording critical phase...[/cyan]")
            if key in (ord("s"), ord("f")) and recording:
                success = key == ord("s")
                meta = EpisodeMetadata(
                    task="plug_insertion",
                    language=language,
                    phase=dc.get("phase", "critical"),
                    success=success,
                    robot=f"fr3_{gripper_type}",
                )
                if not buffers["timestamps"]:
                    console.print("[red]No frames recorded — discarding[/red]")
                    recording = False
                    continue
                ep = CriticalPhaseEpisode(
                    metadata=meta,
                    timestamps=np.array(buffers["timestamps"], dtype=np.float64),
                    proprio=np.stack(buffers["proprio"]),
                    actions=np.stack(buffers["actions"]),
                    rewards=np.array(buffers["rewards"], dtype=np.float32),
                    dones=np.array(buffers["dones"], dtype=bool),
                    is_human=np.ones(len(buffers["timestamps"]), dtype=bool),
                    images_wrist=np.stack(buffers["wrist"]) if buffers["wrist"] else None,
                    images_external=np.stack(buffers["external"]) if buffers["external"] else None,
                )
                path = save_episode(ep, episodes_dir)
                episode_count += 1
                console.print(
                    f"[green]Saved #{episode_count}[/green] {path.name} steps={ep.num_steps()} success={success}"
                )
                recording = False

            if not recording:
                time.sleep(0.001)
                continue
            if len(robot._state_buffer) == 0:
                time.sleep(dt)
                continue

            width = gripper.position
            proprio = _proprio_from_state(robot, width)
            action_norm = _normalize_action(action_raw, pos_scale, rot_scale)

            wrist_img, ext_img = None, None
            if camera_manager is not None:
                frames = camera_manager.get_all_latest_frames()
                if "wrist" in frames:
                    wrist_img = cv2.cvtColor(
                        cv2.resize(frames["wrist"]["rgb"], img_size), cv2.COLOR_BGR2RGB
                    )
                if "external" in frames:
                    ext_img = cv2.cvtColor(
                        cv2.resize(frames["external"]["rgb"], img_size), cv2.COLOR_BGR2RGB
                    )

            buffers["timestamps"].append(time.time() - t0)
            buffers["proprio"].append(proprio)
            buffers["actions"].append(action_norm)
            buffers["rewards"].append(1.0 if key == ord("s") else 0.0)
            buffers["dones"].append(False)
            if wrist_img is not None:
                buffers["wrist"].append(wrist_img)
            if ext_img is not None:
                buffers["external"].append(ext_img)

            time.sleep(dt)
    finally:
        gripper.cleanup()
        robot.close()
        if camera_manager is not None:
            camera_manager.close()
        cv2.destroyAllWindows()
        console.print(f"Done. {episode_count} episodes in {episodes_dir}")


if __name__ == "__main__":
    main()
