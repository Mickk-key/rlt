#!/usr/bin/env python3
"""Quick Robotiq connectivity test using RLT local driver."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from rlt.hardware.robotiq_gripper import RobotiqConfig, RobotiqGripperWrapper


def main() -> int:
    parser = argparse.ArgumentParser(description="Robotiq gripper hardware check")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "configs" / "robotiq.yaml"),
    )
    parser.add_argument("--port", type=str, default=None, help="Override serial port")
    args = parser.parse_args()

    cfg = RobotiqConfig.from_yaml(args.config)
    if args.port:
        cfg.serial_port = args.port

    port = cfg.resolve_port()
    print(f"[INFO] Connecting to Robotiq on {port} ...")
    gripper = RobotiqGripperWrapper(cfg)
    print("[INFO] Gripper activated successfully.")
    print(f"[INFO] Current position: {gripper.position:.4f} m")

    print("[INFO] Opening gripper ...")
    gripper.open()
    time.sleep(2.0)
    print(
        f"  position after open: {gripper.position:.4f} m, is_open={gripper.is_open}"
    )

    print("[INFO] Closing gripper ...")
    gripper.close()
    time.sleep(2.0)
    print(
        f"  position after close: {gripper.position:.4f} m, is_open={gripper.is_open}"
    )

    gripper.cleanup()
    print("[INFO] Robotiq gripper check completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
