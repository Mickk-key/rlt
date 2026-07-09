#!/usr/bin/env python3
"""Reset arm to SFT/collection initial pose (fast direct move, optional random cube sample)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml
from rich.console import Console

from rlt.hardware.deoxys.collection_reset import reset_to_init_cube
from rlt.hardware.deoxys.fast_reset import InitCubeConfig
from rlt.hardware.gripper_factory import create_gripper, uses_deoxys_gripper
from rlt.util.deoxys_paths import resolve_controller_cfg_path, smq_root_from_rlt

console = Console()


def _setup_deoxys(deoxys_root: str):
    root = Path(deoxys_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from deoxys import config_root
    from deoxys.franka_interface import FrankaInterface
    from deoxys.utils import YamlConfig
    from deoxys.utils.config_utils import get_default_controller_config

    return config_root, FrankaInterface, YamlConfig, get_default_controller_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast reset to collection initial EE pose")
    parser.add_argument("--config", type=Path, default=Path("configs/sft_plug_insertion.yaml"))
    parser.add_argument("--interface-cfg", type=str, default=None)
    parser.add_argument("--fixed", action="store_true", help="Bottom center only (no xy random)")
    parser.add_argument("--random", action="store_true", help="Random sample in init cube (default)")
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    robot_cfg = raw["robot"]
    sc = raw.get("sft_collection", {})
    dc = raw.get("data_collection", {})
    ws_raw = sc.get("workspace_randomization", {})
    InitCubeConfig.from_yaml_dict(ws_raw)  # validate

    deoxys_root = robot_cfg["deoxys_root"]
    interface_cfg = args.interface_cfg or robot_cfg["deoxys_config"]
    fps = float(sc.get("fps", 50))

    config_root, FrankaInterface, YamlConfig, _ = _setup_deoxys(deoxys_root)
    if not interface_cfg.startswith("/"):
        interface_path = os.path.join(config_root, Path(interface_cfg).name)
        if not os.path.isfile(interface_path):
            interface_path = interface_cfg
    else:
        interface_path = interface_cfg

    has_gripper = uses_deoxys_gripper(raw)
    robot = FrankaInterface(
        interface_path,
        control_freq=fps,
        has_gripper=has_gripper,
        automatic_gripper_reset=False,
    )
    gripper = create_gripper(raw, robot_interface=robot)
    smq_root = smq_root_from_rlt()
    joint_cfg = YamlConfig(
        str(
            resolve_controller_cfg_path(
                "joint-position-controller.yml",
                smq_root=smq_root,
                deoxys_config_root=config_root,
            )
        )
    ).as_easydict()
    osc_cfg = YamlConfig(
        str(
            resolve_controller_cfg_path(
                "configs/deoxys/osc-position-controller.yml",
                smq_root=smq_root,
                deoxys_config_root=config_root,
            )
        )
    ).as_easydict()

    randomize = not args.fixed
    if args.random:
        randomize = True

    try:
        result = reset_to_init_cube(
            robot,
            gripper=gripper,
            osc_position_cfg=osc_cfg,
            joint_controller_cfg=joint_cfg,
            raw=raw,
            randomize=randomize,
            logger=console,
        )
        console.print(
            f"[green]Reset OK[/green] target={result.target_xyz.round(4).tolist()} "
            f"steps={result.steps} err={result.pos_err_m*100:.2f}cm"
        )
    finally:
        gripper.cleanup()
        robot.close()


if __name__ == "__main__":
    main()
