#!/usr/bin/env python3
"""Collect plug-insertion critical-phase demos: deoxys arm + gripper + RealSense.

Designed for RLT reproduction without GPU on the robot PC.
Save NPZ episodes here; copy `data/episodes/plug_insertion/` to the GPU host for training.

Controls (OpenCV window `rlt_collect` must be focused for keys):
  r  start recording (enter critical phase — aligned, about to insert)
  s  save episode as SUCCESS
  f  save episode as FAILURE
  q  quit

SpaceMouse: translate/rotate arm.
  LEFT (once) or g: latch gripper closed; o opens gripper manually.
  RIGHT: reset arm to home joints (program keeps running).
  r/s/f/q: recording controls (recording does NOT start automatically).
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import tty
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import cv2
import numpy as np
import yaml
from rich.console import Console

from rlt.data.episode_io import count_episodes, next_episode_index, save_episode
from rlt.data.schema import CriticalPhaseEpisode, EpisodeMetadata
from rlt.hardware.deoxys_arm import o_t_ee_to_pose
from rlt.hardware.gripper_factory import create_gripper, uses_deoxys_gripper
from rlt.teleop.spacemouse_control import (
    DEFAULT_RESET_JOINTS,
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


def _proprio_from_state(robot_interface, gripper_width: float) -> np.ndarray:
    st = robot_interface._state_buffer[-1]
    pos, quat = o_t_ee_to_pose(st.O_T_EE)
    return np.concatenate([pos, quat, [gripper_width]]).astype(np.float32)


def _normalize_action(action7: np.ndarray, pos_scale: float, rot_scale: float) -> np.ndarray:
    out = action7.copy()
    out[:3] *= pos_scale
    out[3:6] *= rot_scale
    return out.astype(np.float32)


@contextmanager
def _terminal_keys():
    """Non-blocking single-key reads from the collecting terminal (SSH-safe)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _poll_terminal_key() -> int:
    if not select.select([sys.stdin], [], [], 0)[0]:
        return 0
    ch = sys.stdin.read(1)
    return ord(ch) if ch else 0


def _wait_cameras(camera_manager, names: list[str], timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        frames = camera_manager.get_all_latest_frames()
        if all(n in frames for n in names):
            return
        time.sleep(0.05)
    missing = [n for n in names if n not in camera_manager.get_all_latest_frames()]
    raise RuntimeError(f"Camera startup failed (missing frames): {missing}")


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
    cam_fps = int(cam_cfg.get("fps", 30))
    dt = 1.0 / fps
    episodes_dir = Path(paths["episodes_dir"])
    episodes_dir.mkdir(parents=True, exist_ok=True)

    deoxys_root = robot_cfg["deoxys_root"]
    interface_cfg = args.interface_cfg or robot_cfg["deoxys_config"]
    controller_type = dc.get("controller_type", "OSC_POSE")
    controller_cfg_name = dc.get("controller_cfg", "osc-position-controller.yml")
    gripper_latch = bool(dc.get("gripper_latch", True))
    reset_joints = list(dc.get("reset_joint_positions") or DEFAULT_RESET_JOINTS)
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

    device = SpaceMouse(
        vendor_id=int(dc.get("spacemouse_vendor_id", 9583)),
        product_id=int(dc.get("spacemouse_product_id", 50746)),
    )
    device.start_control()

    camera_mapping = cam_cfg.get("mapping", {})
    camera_manager = None
    cam_w = int(cam_cfg.get("width", 640))
    cam_h = int(cam_cfg.get("height", 480))
    if camera_mapping:
        camera_manager = RealSenseCameraManager(
            camera_name_mapping=camera_mapping,
            enable_depth=False,
            fps=cam_fps,
            rgb_resolution=(cam_w, cam_h),
            depth_resolution=(cam_w, cam_h),
        )
        camera_manager.start()
        _wait_cameras(camera_manager, list(camera_mapping.keys()))
        console.print(f"Cameras OK @ {cam_fps}fps: {camera_mapping}")

    img_size = tuple(cam_cfg.get("image_size", [224, 224]))
    existing = count_episodes(episodes_dir)
    next_idx = next_episode_index(episodes_dir)
    console.print(f"[bold]Plug insertion collection[/bold] -> {episodes_dir}")
    if existing:
        console.print(
            f"[cyan]Existing episodes: {existing}[/cyan] — new saves continue from "
            f"[bold]ep_{next_idx:05d}[/bold] (will not overwrite)"
        )
    else:
        console.print("[cyan]No existing episodes — saving from ep_00000[/cyan]")
    console.print(f"Gripper backend: {gripper_type}")
    if gripper_latch and has_deoxys_gripper:
        console.print(
            "[yellow]Gripper:[/yellow] LEFT once or [cyan]g[/cyan] → latch closed. "
            "[cyan]o[/cyan] → open manually."
        )
    else:
        console.print("SpaceMouse: LEFT hold = close, release = open")
    console.print("[yellow]SpaceMouse RIGHT[/yellow] → arm home pose (stay in session)")
    console.print("[bold]Recording is manual — NOT auto on startup[/bold]")
    console.print("  [cyan]r[/cyan] = START recording (critical phase)")
    console.print("  [green]s[/green] = STOP + save SUCCESS  |  [red]f[/red] = STOP + save FAIL")
    console.print("  [yellow]q[/yellow] = quit (no save)")
    console.print("Record ONLY the critical insertion phase (last ~1-3 s before seated).")

    use_gui = bool(os.environ.get("DISPLAY"))
    if use_gui:
        cv2.namedWindow("rlt_collect", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("rlt_collect", 320, 120)
        placeholder = np.zeros((120, 320, 3), dtype=np.uint8)
        cv2.putText(
            placeholder, "focus here for keys", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
    else:
        key_hint = "r/s/f/g/o/q" if gripper_latch else "r/s/f/q"
        console.print(f"[yellow]No DISPLAY — press {key_hint} in THIS terminal[/yellow]")
        placeholder = None

    recording = False
    buffers: dict[str, list] = {
        k: [] for k in ["timestamps", "proprio", "actions", "rewards", "dones", "is_human", "wrist", "external"]
    }
    t0 = 0.0
    session_saved = 0
    gripper_latched = False

    key_ctx = _terminal_keys() if not use_gui else nullcontext()

    try:
        with key_ctx:
            while True:
                action_raw, grasp = input2action(device=device, controller_type=controller_type)
                if is_spacemouse_reset(action_raw):
                    if recording:
                        console.print("[yellow]Reset — discarding in-progress recording[/yellow]")
                        recording = False
                    console.print("[cyan]SpaceMouse reset → moving to home joints...[/cyan]")
                    move_arm_to_reset_pose(
                        robot,
                        reset_joints,
                        controller_cfg=joint_reset_cfg,
                        gripper_open=False,
                    )
                    continue

                action_raw = np.asarray(action_raw, dtype=np.float32)
                if controller_type == "OSC_POSITION":
                    action_raw[3:6] = 0.0

                if use_gui:
                    key = cv2.waitKey(1) & 0xFF
                    cv2.imshow("rlt_collect", placeholder)
                else:
                    key = _poll_terminal_key()

                if key == ord("g"):
                    gripper_latched = True
                if key == ord("o") and has_deoxys_gripper:
                    gripper_latched = False
                    open_franka_gripper(robot, hold_sec=0.3)

                action_raw, gripper_latched = apply_gripper_latch(
                    action_raw,
                    grasp_pressed=bool(grasp),
                    latched=gripper_latched,
                    enabled=gripper_latch and has_deoxys_gripper,
                )

                robot.control(controller_type=controller_type, action=action_raw, controller_cfg=controller_cfg)
                if not has_deoxys_gripper:
                    gripper.apply_action(float(action_raw[-1]))

                if key == ord("q"):
                    break
                if key == ord("r") and not recording:
                    if gripper_latch and has_deoxys_gripper and not gripper_latched:
                        console.print("[yellow]Tip: LEFT or g to grasp plug first[/yellow]")
                    recording = True
                    buffers = {k: [] for k in buffers}
                    t0 = time.time()
                    console.print("[cyan]▶ Recording started — press s (success) or f (fail) to save & stop[/cyan]")
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
                    session_saved += 1
                    console.print(
                        f"[green]Saved {path.name}[/green] "
                        f"(session {session_saved}, total {existing + session_saved}) "
                        f"steps={ep.num_steps()} success={success}"
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
            camera_manager.stop()
        if use_gui:
            cv2.destroyAllWindows()
        total = count_episodes(episodes_dir)
        console.print(f"Done. Saved {session_saved} this session ({total} total in {episodes_dir})")


if __name__ == "__main__":
    main()
