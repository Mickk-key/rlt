#!/usr/bin/env python3
"""SFT / imitation-learning demo collection for plug insertion (JSONL + images).

Franka 7D delta EE control + RealSense wrist/front cameras @ synced timestamps.
Each episode: random EE xy in 10 cm × 10 cm workspace region, 50 Hz control loop.

Controls (terminal or OpenCV window):
  r  reset (+ start recording unless --mock)
  s  STOP + save SUCCESS (recording mode only)
  f  STOP + save FAILURE (recording mode only)
  q  quit

Mock mode (--mock / MOCK=1): r = fast reset + teleop only, no disk writes.

SpaceMouse:
  translate/rotate arm | LEFT/g latch gripper | o open | RIGHT joint home (no exit)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import yaml
from rich.console import Console

from rlt.data.sft_io import count_sft_episodes, finalize_sft_episode, next_sft_episode_index, open_sft_episode
from rlt.data.sft_recorder import SFTRecorder, SFTRecorderConfig, normalize_action, proprio_from_state
from rlt.data.sync_sampler import SyncError, RobotFrame
from rlt.data.sft_recorder import robot_state_timestamp
from rlt.hardware.gripper_factory import create_gripper, uses_deoxys_gripper
from rlt.hardware.deoxys.fast_reset import InitCubeConfig
from rlt.hardware.workspace_reset import reset_random_workspace
from rlt.util.deoxys_paths import default_osc_controller_cfg_name, resolve_controller_cfg_path, smq_root_from_rlt
from rlt.teleop.spacemouse_control import (
    DEFAULT_RESET_JOINTS,
    acknowledge_spacemouse_reset,
    apply_gripper_latch,
    is_spacemouse_reset,
    move_arm_to_reset_pose,
    open_franka_gripper,
)
from rlt.util.terminal_keys import poll_key, terminal_keys

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


def _wait_cameras(camera_manager, names: list[str], timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        frames = camera_manager.get_all_latest_frames()
        if all(n in frames for n in names):
            return
        time.sleep(0.05)
    missing = [n for n in names if n not in camera_manager.get_all_latest_frames()]
    raise RuntimeError(f"Camera startup failed (missing frames): {missing}")


def _load_workspace_cfg(raw: dict) -> InitCubeConfig:
    sc = raw.get("sft_collection", {})
    return InitCubeConfig.from_yaml_dict(sc.get("workspace_randomization", {}))


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT plug-insertion demo collector (JSONL + PNG)")
    parser.add_argument("--config", type=Path, default=Path("configs/sft_plug_insertion.yaml"))
    parser.add_argument("--interface-cfg", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate config and exit")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Practice mode: r=reset+teleop only, no JSONL/PNG writes",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    robot_cfg = raw["robot"]
    sc = raw.get("sft_collection", {})
    dc = raw.get("data_collection", {})
    cam_cfg = raw["cameras"]
    paths = raw["paths"]
    language = raw["vla"]["language_instruction"]
    gripper_type = raw.get("gripper", {}).get("type", "franka")

    fps = float(sc.get("fps", 50))
    cam_fps = int(sc.get("camera_fps", cam_cfg.get("fps", 30)))
    dt = 1.0 / fps
    episodes_dir = Path(sc.get("output_dir", paths.get("sft_episodes_dir", "data/sft/plug_insertion")))
    episodes_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        console.print(f"[green]Config OK[/green] fps={fps} output={episodes_dir}")
        return

    deoxys_root = robot_cfg["deoxys_root"]
    interface_cfg = args.interface_cfg or robot_cfg["deoxys_config"]
    controller_type = sc.get("controller_type", dc.get("controller_type", "OSC_POSE"))
    controller_cfg_name = sc.get(
        "controller_cfg",
        dc.get("controller_cfg", default_osc_controller_cfg_name(controller_type)),
    )
    gripper_latch = bool(sc.get("gripper_latch", dc.get("gripper_latch", True)))
    reset_joints = list(sc.get("reset_joint_positions", dc.get("reset_joint_positions")) or DEFAULT_RESET_JOINTS)
    pos_scale, rot_scale, _ = robot_cfg.get("action_scale", [0.05, 0.02, 1.0])
    ws_cfg = _load_workspace_cfg(raw)

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

    smq_root = smq_root_from_rlt()
    controller_cfg = get_default_controller_config(controller_type)
    if controller_cfg_name:
        cfg_path = resolve_controller_cfg_path(
            controller_cfg_name, smq_root=smq_root, deoxys_config_root=config_root
        )
        controller_cfg = YamlConfig(str(cfg_path)).as_easydict()
    joint_reset_cfg = YamlConfig(
        str(
            resolve_controller_cfg_path(
                "joint-position-controller.yml",
                smq_root=smq_root,
                deoxys_config_root=config_root,
            )
        )
    ).as_easydict()
    osc_position_cfg = YamlConfig(
        str(
            resolve_controller_cfg_path(
                "configs/deoxys/osc-position-controller.yml",
                smq_root=smq_root,
                deoxys_config_root=config_root,
            )
        )
    ).as_easydict()

    device = SpaceMouse(
        vendor_id=int(sc.get("spacemouse_vendor_id", dc.get("spacemouse_vendor_id", 9583))),
        product_id=int(sc.get("spacemouse_product_id", dc.get("spacemouse_product_id", 50746))),
    )
    device.start_control()

    camera_mapping = cam_cfg.get("mapping", {})
    cam_w = int(cam_cfg.get("width", 640))
    cam_h = int(cam_cfg.get("height", 480))
    camera_manager = RealSenseCameraManager(
        camera_name_mapping=camera_mapping,
        enable_depth=False,
        fps=cam_fps,
        rgb_resolution=(cam_w, cam_h),
        depth_resolution=(cam_w, cam_h),
    )
    camera_manager.start()
    _wait_cameras(camera_manager, list(camera_mapping.keys()))

    img_size = tuple(sc.get("image_size", cam_cfg.get("image_size", [cam_w, cam_h])))
    post_reset_warmup_steps = int(sc.get("post_reset_warmup_steps", 50))
    recorder_cfg = SFTRecorderConfig(
        fps=fps,
        max_sync_delta_ms=float(sc.get("max_sync_delta_ms", 50.0)),
        max_sync_retries=int(sc.get("max_sync_retries", 8)),
        sync_retry_sleep_sec=float(sc.get("sync_retry_sleep_sec", 0.002)),
        image_size=(int(img_size[0]), int(img_size[1])),
        pos_scale=float(pos_scale),
        rot_scale=float(rot_scale),
        camera_front_key=sc.get("camera_front_key", "external"),
        camera_wrist_key=sc.get("camera_wrist_key", "wrist"),
        post_reset_warmup_steps=post_reset_warmup_steps,
    )
    recorder = SFTRecorder(
        robot=robot,
        gripper=gripper,
        camera_manager=camera_manager,
        camera_mapping=camera_mapping,
        cfg=recorder_cfg,
    )

    existing = count_sft_episodes(episodes_dir)
    next_idx = next_sft_episode_index(episodes_dir)
    mock_mode = args.mock
    console.print(f"[bold]SFT plug insertion collection[/bold] → {episodes_dir}")
    if mock_mode:
        console.print("[yellow bold]MOCK 模式[/yellow bold] — 只 reset + 遥操，不写盘。确认无误后去掉 --mock 再采。")
    console.print(f"Control {fps} Hz | cameras {cam_fps} Hz | sync ≤ {recorder_cfg.max_sync_delta_ms} ms")
    console.print(f"Post-reset warmup: {post_reset_warmup_steps} steps (~{post_reset_warmup_steps / fps:.1f}s before first frame)")
    console.print(f"Init cube bottom_center={ws_cfg.bottom_center_xyz} ±{ws_cfg.xy_half_range_m*100:.0f}cm xy")
    if mock_mode:
        console.print("[cyan]r[/cyan]=reset+试遥操  [yellow]q[/yellow]=quit  (s/f 在 mock 下无效)")
    else:
        console.print("[cyan]r[/cyan]=reset+record  [green]s[/green]=save OK  [red]f[/red]=save fail  [yellow]q[/yellow]=quit")
    console.print("SpaceMouse RIGHT → joint home (stay in session)")

    use_gui = bool(os.environ.get("DISPLAY"))
    if use_gui:
        cv2.namedWindow("sft_collect", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("sft_collect", 320, 120)
        placeholder = np.zeros((120, 320, 3), dtype=np.uint8)
        cv2.putText(placeholder, "focus for keys", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        console.print("[yellow]No DISPLAY — use this terminal for r/s/f/q/g/o[/yellow]")
        placeholder = None

    recording = False
    episode_dir = None
    episode_meta = None
    episode_t0 = 0.0
    step = 0
    dropped = 0
    max_sync = 0.0
    session_saved = 0
    gripper_latched = False
    last_reset_offset = [0.0, 0.0]
    last_reset_xyz = list(ws_cfg.bottom_center_xyz)
    mock_reset_count = 0
    pending_success = True
    stop_recording = False
    warmup_steps_remaining = 0

    key_ctx = terminal_keys() if not use_gui else nullcontext()

    try:
        with key_ctx:
            while True:
                loop_start = time.time()
                action_raw, grasp = input2action(device=device, controller_type=controller_type)
                if is_spacemouse_reset(action_raw):
                    if recording:
                        console.print("[yellow]Reset — discarding in-progress episode[/yellow]")
                        recording = False
                        episode_dir = None
                    console.print("[cyan]SpaceMouse → joint home[/cyan]")
                    move_arm_to_reset_pose(robot, reset_joints, controller_cfg=joint_reset_cfg, gripper_open=False)
                    acknowledge_spacemouse_reset(device)
                    continue

                action_raw = np.asarray(action_raw, dtype=np.float32)
                if controller_type == "OSC_POSITION":
                    action_raw[3:6] = 0.0

                if use_gui:
                    key = cv2.waitKey(1) & 0xFF
                    cv2.imshow("sft_collect", placeholder)
                else:
                    key = poll_key()

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
                    if recording:
                        console.print("[yellow]Quit without saving current episode[/yellow]")
                    break

                if key == ord("r") and not recording:
                    console.print("[cyan]Fast reset to init cube...[/cyan]")
                    ws_result = reset_random_workspace(
                        robot,
                        gripper=gripper,
                        joint_controller_cfg=joint_reset_cfg,
                        osc_position_cfg=osc_position_cfg,
                        ws_cfg=ws_cfg,
                        sft_cfg=sc,
                        pos_tol_m=float(sc.get("reset_pos_tol_m", 0.015)),
                        logger=console,
                    )
                    last_reset_offset = ws_result.offset_xyz[:2].tolist()
                    last_reset_xyz = ws_result.target_xyz.tolist()
                    info = ws_result.reset_info
                    console.print(
                        f"[green]Reset OK[/green] target={last_reset_xyz} "
                        f"offset_xy(cm)={[round(x * 100, 2) for x in last_reset_offset]} "
                        f"steps={info.get('steps')} err={info.get('pos_err_m', 0)*100:.2f}cm"
                    )
                    if mock_mode:
                        mock_reset_count += 1
                        console.print(
                            f"[yellow]MOCK[/yellow] #{mock_reset_count} — 请 SpaceMouse 试遥操。"
                            "满意后退出，用 MOCK=0 或去掉 --mock 正式采集。"
                        )
                        continue
                    episode_dir, episode_meta = open_sft_episode(episodes_dir)
                    episode_meta.language = language
                    episode_meta.fps = fps
                    episode_meta.workspace_offset_xy = last_reset_offset
                    episode_meta.reset_target_xyz = last_reset_xyz
                    episode_meta.extra = {"robot": f"fr3_{gripper_type}", "controller": controller_type}
                    recording = True
                    stop_recording = False
                    step = 0
                    dropped = 0
                    max_sync = 0.0
                    warmup_steps_remaining = post_reset_warmup_steps
                    episode_t0 = time.time()
                    recorder._cache.update_from_manager(camera_manager, mapping=camera_mapping)
                    recorder._prev_cam_snapshot = {}
                    console.print(
                        f"[green]▶ Recording {episode_meta.episode_id}[/green] — "
                        f"warmup {post_reset_warmup_steps} steps, then s=success f=fail"
                    )
                    continue

                if key in (ord("s"), ord("f")) and mock_mode and not recording:
                    console.print("[yellow]MOCK 模式不保存。去掉 --mock 后 r→遥操→s/f 才会写盘。[/yellow]")
                    continue

                if key in (ord("s"), ord("f")) and recording:
                    pending_success = key == ord("s")
                    stop_recording = True

                if recording and not stop_recording:
                    if len(robot._state_buffer) == 0:
                        time.sleep(dt)
                        continue

                    recorder._cache.update_from_manager(camera_manager, mapping=camera_mapping)
                    if warmup_steps_remaining > 0:
                        warmup_steps_remaining -= 1
                        recorder._prev_cam_snapshot = recorder._cache.snapshot()
                        if warmup_steps_remaining == 0:
                            episode_t0 = time.time()
                            console.print("[cyan]Warmup done — recording from next step[/cyan]")
                        continue

                    width = gripper.position
                    proprio = proprio_from_state(robot, width)
                    robot_ts = time.time()
                    robot_frame = RobotFrame(
                        timestamp=robot_ts,
                        proprio=proprio,
                        robot_state_timestamp=robot_state_timestamp(robot),
                    )
                    action_norm = normalize_action(action_raw, pos_scale, rot_scale)

                    synced = None
                    for _ in range(recorder_cfg.max_sync_retries):
                        try:
                            synced = recorder.sync_camera_robot_timestamp(robot_frame)
                            break
                        except SyncError:
                            time.sleep(recorder_cfg.sync_retry_sleep_sec)
                            recorder._cache.update_from_manager(camera_manager, mapping=camera_mapping)

                    if synced is None:
                        dropped += 1
                        if dropped % 10 == 1:
                            console.print(f"[yellow]Sync miss (dropped={dropped})[/yellow]")
                    else:
                        max_sync = max(max_sync, synced.max_sync_delta_ms)
                        if recorder.save_transition(
                            episode_dir,
                            step,
                            robot_frame=robot_frame,
                            synced=synced,
                            action_norm=action_norm,
                            episode_t0=episode_t0,
                        ):
                            recorder._prev_cam_snapshot = recorder._cache.snapshot()
                            step += 1
                        else:
                            dropped += 1
                            recorder._prev_cam_snapshot = recorder._cache.snapshot()

                elif recording and stop_recording:
                    episode_meta.success = pending_success
                    episode_meta.num_steps = step
                    episode_meta.dropped_steps = dropped
                    episode_meta.max_sync_delta_ms = max_sync
                    finalize_sft_episode(episode_dir, episode_meta)
                    session_saved += 1
                    label = "SUCCESS" if pending_success else "FAIL"
                    console.print(
                        f"[green]Saved {episode_meta.episode_id}[/green] {label} "
                        f"steps={step} dropped={dropped} max_sync={max_sync:.1f}ms"
                    )
                    recording = False
                    stop_recording = False
                    episode_dir = None

                elapsed = time.time() - loop_start
                time.sleep(max(0.0, dt - elapsed))
    finally:
        device.close()
        gripper.cleanup()
        robot.close()
        camera_manager.stop()
        if use_gui:
            cv2.destroyAllWindows()
        total = count_sft_episodes(episodes_dir)
        console.print(f"Done. Saved {session_saved} this session ({total} total in {episodes_dir})")


if __name__ == "__main__":
    main()
