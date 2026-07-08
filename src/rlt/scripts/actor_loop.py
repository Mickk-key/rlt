#!/usr/bin/env python3
"""Robot-side actor loop for split online RL (Algorithm 1 rollout only).

Each step: get obs → send to GPU → receive action → execute on Franka → log transition.
Reward and episode termination come from RewardLogger (s/f keys or timeout).
Reset is handled by ResetManager before each episode (not part of RL step).

Usage:
  MOCK=1 python -m rlt.scripts.actor_loop --mock          # no robot, no GPU
  python -m rlt.scripts.actor_loop --config configs/plug_insertion.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rich.console import Console

from rlt.hardware.deoxys.camera_source import build_deoxys_realsense_pair, read_rgb_frames
from rlt.hardware.deoxys.reset_manager import ResetManager
from rlt.hardware.gripper_factory import create_robot_env
from rlt.rl.gpu_client import GPUClient, MockGPUClient, create_gpu_client
from rlt.rl.reward_logger import EpisodeOutcome, RewardLogger
from rlt.rl.ws_protocol import pack_observation
from rlt.sim.mock_env import MockPrecisionEnv
from rlt.util.deoxys_paths import apply_deoxys_paths
from rlt.util.terminal_keys import stdin_is_tty, terminal_keys

console = Console()


@dataclass
class StepRecord:
    step: int
    proprio: list[float]
    action: list[float]
    reward: float
    done: bool


def _setup_cameras(raw: dict, robot_cfg: dict):
    cam_cfg = raw.get("cameras", {})
    backend = cam_cfg.get("backend", "deoxys_realsense")
    if backend != "deoxys_realsense":
        raise ValueError(f"Unsupported cameras.backend={backend!r}; use deoxys_realsense")
    return build_deoxys_realsense_pair(cam_cfg, deoxys_root=robot_cfg.get("deoxys_root"))


def _build_observation(
    proprio: np.ndarray,
    camera_manager,
    camera_mapping: dict[str, str],
    language: str,
) -> dict[str, Any]:
    obs: dict[str, Any] = {
        "proprio": proprio.astype(np.float32),
        "language": language,
    }
    images = read_rgb_frames(camera_manager, camera_mapping)
    if images:
        obs["images"] = images
    return obs


def _write_transition_log(log_dir: Path, episode_id: str, records: list[StepRecord]) -> Path:
    out_dir = log_dir / "transitions"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{episode_id}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec)) + "\n")
    return path


def run_episode(
    env,
    gpu: GPUClient,
    reward_logger: RewardLogger,
    *,
    chunk_length: int,
    execute_prefix: int,
    control_hz: float,
    language: str,
    camera_manager,
    camera_mapping: dict[str, str],
    log_dir: Path | None,
    episode_index: int,
) -> EpisodeOutcome:
    """One critical-phase episode: infer chunks, execute, poll reward until done."""
    dt = 1.0 / control_hz
    proprio = env.get_proprio() if hasattr(env, "get_proprio") else env.reset()
    step = 0
    total_reward = 0.0
    records: list[StepRecord] = []
    episode_id = f"ep_{episode_index:04d}"

    while True:
        obs = _build_observation(proprio, camera_manager, camera_mapping, language)
        result = gpu.infer(obs)
        if result.state is None:
            raise RuntimeError("GPU infer missing encoded state — restart GPU rl_server with latest code")
        current_state = np.asarray(result.state, dtype=np.float32)
        action_chunk = result.action_chunk
        ref_chunk = result.reference_action
        if step == 0 and result.meta:
            console.print(f"[dim]GPU infer meta[/dim] {result.meta}")
            console.print(
                f"[dim]first action[/dim] pos={action_chunk[0][:3].round(4).tolist()} "
                f"gripper_raw={float(action_chunk[0][6]):.4f} (latched→1.0 if gripper_latch)"
            )
        n_exec = min(chunk_length, execute_prefix, len(action_chunk))

        for i in range(n_exec):
            action = action_chunk[i].astype(np.float32)
            ref_step = ref_chunk[i].astype(np.float32)
            next_proprio, _, _, _ = env.step(action)
            step += 1

            outcome = reward_logger.poll(step)
            step_reward = float(outcome.reward) if outcome else 0.0
            step_done = bool(outcome.done) if outcome else False
            total_reward += step_reward

            records.append(
                StepRecord(
                    step=step,
                    proprio=proprio.tolist(),
                    action=action.tolist(),
                    reward=step_reward,
                    done=step_done,
                )
            )

            next_obs = _build_observation(next_proprio, camera_manager, camera_mapping, language)
            trans: dict[str, Any] = {
                "state": current_state.tolist(),
                "action": action.tolist(),
                "reference_action": ref_step.tolist(),
                "reward": step_reward,
                "done": step_done,
                "next_proprio": next_proprio.astype(np.float32).tolist(),
                "language": language,
            }
            if next_obs.get("images"):
                packed = pack_observation(
                    next_proprio.astype(np.float32),
                    images=next_obs["images"],
                    language=language,
                )
                if "images_jpeg" in packed:
                    trans["next_images_jpeg"] = packed["images_jpeg"]

            t_resp = gpu.send_transition(trans)
            current_state = np.asarray(t_resp["next_state"], dtype=np.float32)
            if t_resp.get("updated") and t_resp.get("metrics"):
                console.print(
                    f"[dim]learner[/dim] buffer={t_resp.get('buffer_size')} "
                    f"metrics={t_resp.get('metrics')}"
                )

            proprio = next_proprio
            if outcome:
                if log_dir:
                    _write_transition_log(log_dir, episode_id, records)
                    reward_logger.log_episode(
                        outcome,
                        episode_id=episode_id,
                        steps=step,
                        total_reward=total_reward,
                    )
                console.print(
                    f"[green]Episode {episode_id} done[/green] "
                    f"reason={outcome.reason} reward={outcome.reward} steps={step}"
                )
                return outcome

            time.sleep(dt)

        outcome = reward_logger.poll(step)
        if outcome:
            if log_dir:
                _write_transition_log(log_dir, episode_id, records)
                reward_logger.log_episode(
                    outcome,
                    episode_id=episode_id,
                    steps=step,
                    total_reward=total_reward,
                )
            return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot-side actor loop for split online RL")
    parser.add_argument("--config", type=Path, default=Path("configs/plug_insertion.yaml"))
    parser.add_argument("--mock", action="store_true", help="Mock env + mock GPU (no robot)")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--gpu-host", default=None, help="Override gpu_server.host / GPU_SERVER_HOST")
    parser.add_argument("--reset-mode", default=None, choices=["demo", "home", "none"])
    parser.add_argument("--no-cameras", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    apply_deoxys_paths(raw, smq_root=Path(__file__).resolve().parents[4])

    rl = raw.get("online_rl", {})
    vla = raw.get("vla", {})
    paths = raw.get("paths", {})
    chunk_length = rl.get("chunk_length", 10)
    execute_prefix = vla.get("execute_prefix", 20)
    control_hz = rl.get("control_hz", raw.get("robot", {}).get("control_hz", 20))
    max_steps = args.max_steps or int(os.environ.get("MAX_STEPS", rl.get("max_steps_per_episode", 200)))
    episodes = args.episodes or int(os.environ.get("EPISODES", rl.get("max_episodes", 10)))
    language = vla.get("language_instruction", "")

    rlt_root = Path(__file__).resolve().parents[3]
    log_dir = Path(paths.get("log_dir", "logs"))
    if not log_dir.is_absolute():
        log_dir = (rlt_root / log_dir).resolve()
    log_dir = log_dir / "online_rl"

    reward_logger = RewardLogger(
        max_steps=max_steps,
        log_dir=log_dir,
        poll_keys=not args.mock and stdin_is_tty(),
    )
    if not args.mock and not stdin_is_tty():
        console.print("[yellow]stdin 非 TTY — 按键 reward 不可用，仅 timeout 终止[/yellow]")
    env_mock = args.mock
    gpu_cfg = raw.get("gpu_server", {})
    gpu_host = args.gpu_host or os.environ.get("GPU_SERVER_HOST") or gpu_cfg.get("host")
    gpu_mock = env_mock and not gpu_host
    if gpu_cfg.get("mock") is False and gpu_host:
        gpu_mock = False
    if os.environ.get("GPU_SERVER_MOCK", "").lower() in ("0", "false", "no"):
        gpu_mock = False
    if os.environ.get("GPU_SERVER_MOCK", "").lower() in ("1", "true", "yes"):
        gpu_mock = True

    gpu = create_gpu_client(raw, mock=gpu_mock, host_override=gpu_host)
    if gpu_host and not gpu_mock:
        try:
            pong = gpu.ping()
            console.print(f"[cyan]GPU server[/cyan] {pong}")
        except Exception as exc:
            console.print(f"[red]GPU server unreachable[/red]: {exc}")
            console.print("[yellow]Start GPU host: bash scripts/run_rl_server.sh[/yellow]")
            raise

    camera_manager = None
    camera_mapping: dict[str, str] = {}

    if env_mock:
        proprio_dim = raw["robot"]["proprio_dim"]
        action_dim = rl.get("action_dim", 7)

        class MockWrap:
            def __init__(self):
                self._env = MockPrecisionEnv(proprio_dim, action_dim, chunk_length)

            def reset(self):
                return np.zeros(proprio_dim, dtype=np.float32)

            def get_proprio(self):
                return np.zeros(proprio_dim, dtype=np.float32)

            def step(self, a):
                _, r, d, _ = self._env.step(a)
                return np.zeros(proprio_dim, dtype=np.float32), r, d, {}

            def close(self):
                pass

        env = MockWrap()
        tag = "mock env"
        if isinstance(gpu, MockGPUClient):
            tag += " + mock GPU"
        else:
            tag += f" + GPU ws://{gpu_host}"
        console.print(f"[yellow]{tag}[/yellow]")
    else:
        env = create_robot_env(raw, rlt_root=rlt_root)
        if not args.no_cameras:
            camera_manager, camera_mapping = _setup_cameras(raw, raw["robot"])
        if isinstance(gpu, MockGPUClient):
            console.print("[green]Real robot env[/green] [dim](mock GPU — zero actions)[/dim]")
        else:
            console.print(f"[green]Real robot env[/green] [cyan]GPU ws://{gpu_host}[/cyan]")

    reset_manager = ResetManager.from_config(env, raw) if not env_mock else None
    if reset_manager is not None and args.reset_mode:
        from rlt.hardware.deoxys.reset_manager import ResetMode

        reset_manager.mode = ResetMode(args.reset_mode)
    if reset_manager is not None:
        console.print(
            f"[cyan]Reset mode[/cyan]: {reset_manager.mode.value} "
            f"(demo → home joints then random critical pose from data/plug_insertion)"
        )

    console.print(
        "Controls: [green]s[/green]=success (reward=1)  "
        "[red]f[/red]=fail  [yellow]q[/yellow]=quit  "
        f"timeout={max_steps} steps"
    )

    key_ctx = nullcontext() if (env_mock or not stdin_is_tty()) else terminal_keys()
    try:
        with key_ctx:
            for ep in range(episodes):
                if reset_manager is not None:
                    proprio, reset_info = reset_manager.reset()
                    console.print(f"Reset ep {ep}: {reset_info}")
                elif hasattr(env, "reset"):
                    env.reset()

                run_episode(
                    env,
                    gpu,
                    reward_logger,
                    chunk_length=chunk_length,
                    execute_prefix=execute_prefix,
                    control_hz=control_hz,
                    language=language,
                    camera_manager=camera_manager,
                    camera_mapping=camera_mapping,
                    log_dir=log_dir,
                    episode_index=ep,
                )
    finally:
        if camera_manager is not None:
            camera_manager.stop()
        if hasattr(env, "close"):
            env.close()
        gpu.close()


if __name__ == "__main__":
    main()
