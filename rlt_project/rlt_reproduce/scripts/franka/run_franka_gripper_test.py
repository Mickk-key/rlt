#!/usr/bin/env python3
"""Quick Franka Hand connectivity test via deoxys FrankaInterface."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Franka Hand gripper hardware check")
    parser.add_argument(
        "--interface-cfg",
        type=str,
        default="/home/host5010/workspaces/wty/deoxys_control/deoxys/config/charmander.yml",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only verify state streaming (no open/close motion)",
    )
    args = parser.parse_args()

    deoxys_root = Path("/home/host5010/workspaces/wty/deoxys_control/deoxys")
    sys.path.insert(0, str(deoxys_root))
    from deoxys.franka_interface import FrankaInterface

    print(f"[INFO] Connecting deoxys FrankaInterface (has_gripper=True) ...")
    robot = FrankaInterface(
        args.interface_cfg,
        control_freq=20,
        has_gripper=True,
        automatic_gripper_reset=False,
    )

    width = None
    for _ in range(80):
        if robot.received_states and robot.gripper_state_buffer_size > 0:
            width = float(robot.last_gripper_q.reshape(-1)[0])
            break
        time.sleep(0.1)

    if width is None:
        print("[FAIL] No gripper state received.")
        print("       Ensure Desk FCI is active and gripper-interface is running:")
        print("       bash scripts/franka/start_gripper.sh")
        robot.close()
        return 1

    print(f"[ OK ] Franka Hand detected. width={width:.4f} m")

    if args.quick:
        robot.close()
        return 0

    print("[INFO] Sending open command (action=-1) ...")
    robot.gripper_control(-1.0)
    time.sleep(2.0)
    print(f"  width after open: {float(robot.last_gripper_q.reshape(-1)[0]):.4f} m")

    print("[INFO] Sending grasp command (action=+1) ...")
    robot.gripper_control(1.0)
    time.sleep(2.0)
    print(f"  width after grasp: {float(robot.last_gripper_q.reshape(-1)[0]):.4f} m")

    robot.close()
    print("[INFO] Franka Hand check completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
