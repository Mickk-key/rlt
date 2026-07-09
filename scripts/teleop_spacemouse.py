#!/usr/bin/env python3
"""SpaceMouse teleop with smq workspace behavior.

- SpaceMouse RIGHT (reset): move arm to home joints, do NOT exit.
- SpaceMouse LEFT (once) or g: latch gripper closed until o.
- On exit: gripper state is left unchanged (use o to open manually).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from rlt.teleop.spacemouse_control import (
    DEFAULT_RESET_JOINTS,
    acknowledge_spacemouse_reset,
    apply_gripper_latch,
    is_spacemouse_reset,
    move_arm_to_reset_pose,
)


def _setup_deoxys(deoxys_root: str):
    root = Path(deoxys_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from deoxys import config_root
    from deoxys.franka_interface import FrankaInterface
    from deoxys.utils import YamlConfig
    from deoxys.utils.config_utils import get_default_controller_config
    from deoxys.utils.input_utils import input2action
    from deoxys.utils.io_devices import SpaceMouse

    return config_root, FrankaInterface, YamlConfig, get_default_controller_config, input2action, SpaceMouse


def main() -> int:
    parser = argparse.ArgumentParser(description="SpaceMouse teleop (smq workspace)")
    parser.add_argument(
        "--interface-cfg",
        type=str,
        default=os.environ.get("DEOXYS_INTERFACE_CFG", "config/charmander.yml"),
    )
    parser.add_argument("--controller-type", type=str, default=os.environ.get("CONTROLLER_TYPE", "OSC_POSE"))
    parser.add_argument("--controller-cfg", type=str, default="osc-position-controller.yml")
    parser.add_argument("--vendor-id", type=int, default=int(os.environ.get("SPACEMOUSE_VENDOR_ID", "9583")))
    parser.add_argument("--product-id", type=int, default=int(os.environ.get("SPACEMOUSE_PRODUCT_ID", "50746")))
    parser.add_argument(
        "--deoxys-root",
        type=str,
        default="/home/host5010/workspaces/wty/deoxys_control/deoxys",
    )
    parser.add_argument("--gripper-latch", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    (
        config_root,
        FrankaInterface,
        YamlConfig,
        get_default_controller_config,
        input2action,
        SpaceMouse,
    ) = _setup_deoxys(args.deoxys_root)

    if not args.interface_cfg.startswith("/"):
        interface_path = os.path.join(config_root, Path(args.interface_cfg).name)
        if not os.path.isfile(interface_path):
            interface_path = args.interface_cfg
    else:
        interface_path = args.interface_cfg

    device = SpaceMouse(vendor_id=args.vendor_id, product_id=args.product_id)
    device.start_control()

    robot = FrankaInterface(
        interface_path,
        use_visualizer=False,
        has_gripper=True,
        automatic_gripper_reset=False,
    )
    controller_type = args.controller_type
    controller_cfg = YamlConfig(os.path.join(config_root, args.controller_cfg)).as_easydict()
    if controller_cfg.controller_type != controller_type:
        controller_cfg = get_default_controller_config(controller_type)
    joint_reset_cfg = YamlConfig(os.path.join(config_root, "joint-position-controller.yml")).as_easydict()

    reset_joints = DEFAULT_RESET_JOINTS
    gripper_latched = False

    print("SpaceMouse teleop (smq)")
    print("  Move/rotate: SpaceMouse cap")
    print("  LEFT once / g : latch gripper closed")
    print("  o             : open gripper (between trials)")
    print("  RIGHT         : reset arm to home joints (stay in teleop)")
    print("  Ctrl+C        : quit")

    try:
        while True:
            action, grasp = input2action(device=device, controller_type=controller_type)
            if is_spacemouse_reset(action):
                print("[reset] Moving arm to home joints ...")
                move_arm_to_reset_pose(
                    robot,
                    reset_joints,
                    controller_cfg=joint_reset_cfg,
                    gripper_open=False,
                )
                acknowledge_spacemouse_reset(device)
                continue

            action = np.asarray(action, dtype=np.float32)
            if controller_type == "OSC_POSITION":
                action[3:6] = 0.0

            action, gripper_latched = apply_gripper_latch(
                action,
                grasp_pressed=bool(grasp),
                latched=gripper_latched,
                enabled=args.gripper_latch,
            )

            robot.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\n[exit] Stopping teleop ...")
    finally:
        robot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
