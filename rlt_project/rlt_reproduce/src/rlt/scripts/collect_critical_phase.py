#!/usr/bin/env python3
"""Collect critical-phase demonstrations for RLT (teleop or VLA rollout).

Controls:
  SpaceMouse / keyboard teleop: integrate with your existing deoxys teleop script.
  This script records proprio + optional camera frames at fixed Hz.

  s = mark episode success and save
  f = mark failure and save
  q = quit

For full RLT pipeline you need ~15-60 min of critical-phase data per task (paper).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console

from rlt.config import RLTConfig
from rlt.data.episode_io import save_episode
from rlt.data.schema import CriticalPhaseEpisode, EpisodeMetadata
from rlt.hardware.gripper_factory import create_robot_env

console = Console()


def _read_camera(device_id: int, size: tuple[int, int]) -> np.ndarray | None:
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        cap.release()
        return None
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(frame, size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/franka/fr3_franka.yaml"))
    parser.add_argument("--task", type=str, default="precision_insertion")
    parser.add_argument("--backend", type=str, default=None, help="override robot.backend")
    args = parser.parse_args()

    cfg = RLTConfig.from_yaml(args.config)
    cfg.ensure_dirs()
    dc = cfg.__dict__.get("data_collection", {}) if hasattr(cfg, "data_collection") else {}
    # load extra keys from raw yaml
    import yaml

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    dc = raw.get("data_collection", {})
    robot = raw.get("robot", {})
    if args.backend:
        robot["backend"] = args.backend
        raw["robot"] = robot

    fps = float(dc.get("fps", 20))
    dt = 1.0 / fps
    episodes_dir = Path(raw["paths"]["episodes_dir"])
    language = raw.get("vla", {}).get("language_instruction", "")
    img_size = tuple(raw.get("cameras", {}).get("image_size", [224, 224]))
    wrist_id = int(raw.get("cameras", {}).get("wrist_camera_id", 0))
    ext_id = int(raw.get("cameras", {}).get("external_camera_id", 2))

    env = create_robot_env(raw)
    console.print(f"[bold]Collecting[/bold] task={args.task} fps={fps} -> {episodes_dir}")
    console.print("Press [green]s[/green]=success save, [red]f[/red]=fail save, [yellow]q[/yellow]=quit")

    recording = False
    buffers: dict[str, list] = {
        "timestamps": [],
        "proprio": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "is_human": [],
        "wrist": [],
        "external": [],
    }
    t0 = time.time()

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") or key == ord("f"):
                if not recording:
                    recording = True
                    buffers = {k: [] for k in buffers}
                    t0 = time.time()
                    console.print("[cyan]Recording... press s/f again to save[/cyan]")
                    continue
                success = key == ord("s")
                episode = CriticalPhaseEpisode(
                    metadata=EpisodeMetadata(
                        task=args.task,
                        language=language,
                        phase=dc.get("phase", "critical"),
                        success=success,
                    ),
                    timestamps=np.array(buffers["timestamps"], dtype=np.float64),
                    proprio=np.stack(buffers["proprio"]) if buffers["proprio"] else np.zeros((0, 8)),
                    actions=np.stack(buffers["actions"]) if buffers["actions"] else np.zeros((0, 7)),
                    rewards=np.array(buffers["rewards"], dtype=np.float32),
                    dones=np.array(buffers["dones"], dtype=bool),
                    is_human=np.ones(len(buffers["timestamps"]), dtype=bool),
                    images_wrist=np.stack(buffers["wrist"]) if buffers["wrist"] else None,
                    images_external=np.stack(buffers["external"]) if buffers["external"] else None,
                )
                path = save_episode(episode, episodes_dir)
                console.print(f"[green]Saved[/green] {path} success={success} steps={episode.num_steps()}")
                recording = False

            if recording:
                proprio = env.get_proprio()
                # Placeholder zero action — replace with SpaceMouse delta in teleop integration
                action = np.zeros(7, dtype=np.float32)
                buffers["timestamps"].append(time.time() - t0)
                buffers["proprio"].append(proprio)
                buffers["actions"].append(action)
                buffers["rewards"].append(0.0)
                buffers["dones"].append(False)
                w = _read_camera(wrist_id, img_size)
                e = _read_camera(ext_id, img_size)
                if w is not None:
                    buffers["wrist"].append(w)
                if e is not None:
                    buffers["external"].append(e)
            time.sleep(dt)
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
