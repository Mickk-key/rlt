#!/usr/bin/env python3
"""Read current EE pose from Franka and save as init-cube bottom_center in yaml."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from rich.console import Console

from rlt.hardware.deoxys_arm import o_t_ee_to_pose
from rlt.hardware.gripper_factory import create_gripper, uses_deoxys_gripper

console = Console()


def _setup_deoxys(deoxys_root: str):
    root = Path(deoxys_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from deoxys import config_root
    from deoxys.franka_interface import FrankaInterface

    return config_root, FrankaInterface


def read_ee_pose(robot, gripper) -> dict:
    if not robot._state_buffer:
        raise RuntimeError("No robot state — is start_arm.sh running?")
    st = robot._state_buffer[-1]
    pos, quat = o_t_ee_to_pose(st.O_T_EE)
    width = float(gripper.position) if gripper is not None else 0.0
    return {
        "bottom_center_xyz": [float(pos[0]), float(pos[1]), float(pos[2])],
        "reference_quat": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
        "gripper_width": width,
    }


def _update_yaml_workspace(config_path: Path, pose: dict) -> None:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ws = raw.setdefault("sft_collection", {}).setdefault("workspace_randomization", {})
    ws["bottom_center_xyz"] = pose["bottom_center_xyz"]
    ws["reference_quat"] = pose["reference_quat"]
    ws["gripper_width"] = pose["gripper_width"]
    config_path.write_text(yaml.dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record current EE pose as init-cube bottom_center (socket-above reference)"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/sft_plug_insertion.yaml"))
    parser.add_argument("--interface-cfg", type=str, default=None)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Update yaml config (default: print only)",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("data/init_pose"),
        help="Also save JSON snapshot here",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    robot_cfg = raw["robot"]
    deoxys_root = robot_cfg["deoxys_root"]
    interface_cfg = args.interface_cfg or robot_cfg["deoxys_config"]
    fps = float(raw.get("sft_collection", {}).get("fps", 50))

    config_root, FrankaInterface = _setup_deoxys(deoxys_root)
    if not interface_cfg.startswith("/"):
        interface_path = os.path.join(config_root, Path(interface_cfg).name)
        if not os.path.isfile(interface_path):
            interface_path = interface_cfg
    else:
        interface_path = interface_cfg

    console.print("[bold]读取当前末端位姿[/bold]")
    console.print("请先将机械臂遥操到 [green]插座正上方[/green]（采集初始高度），")
    console.print("并 [yellow]停止 teleop[/yellow] 等其他占臂程序，再运行本脚本。")
    console.print("")

    robot = FrankaInterface(
        interface_path,
        control_freq=fps,
        has_gripper=uses_deoxys_gripper(raw),
        automatic_gripper_reset=False,
    )
    gripper = create_gripper(raw, robot_interface=robot)

    try:
        import time

        deadline = time.time() + 10.0
        while time.time() < deadline and not robot._state_buffer:
            time.sleep(0.05)
        pose = read_ee_pose(robot, gripper)
    finally:
        gripper.cleanup()
        robot.close()

    xyz = pose["bottom_center_xyz"]
    quat = pose["reference_quat"]
    console.print(f"  bottom_center_xyz: {xyz}")
    console.print(f"  reference_quat:    {quat}")
    console.print(f"  gripper_width:     {pose['gripper_width']:.6f}")

    args.snapshot_dir.mkdir(parents=True, exist_ok=True)
    snap = {
        **pose,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "note": "Socket-above EE pose = init cube bottom face center",
    }
    snap_path = args.snapshot_dir / "bottom_center.json"
    snap_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    console.print(f"[cyan]Snapshot[/cyan] → {snap_path}")

    if args.write:
        _update_yaml_workspace(config_path, pose)
        console.print(f"[green]Updated[/green] {config_path}")
        rlt_cfg = Path(__file__).resolve().parents[3] / "configs" / config_path.name
        if rlt_cfg.is_file() and rlt_cfg.resolve() != config_path.resolve():
            _update_yaml_workspace(rlt_cfg, pose)
            console.print(f"[green]Updated[/green] {rlt_cfg}")
    else:
        console.print("\n[yellow]Dry-run[/yellow] — add --write to persist to yaml")


if __name__ == "__main__":
    main()
