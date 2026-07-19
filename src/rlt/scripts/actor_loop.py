#!/usr/bin/env python3
"""Robot-side actor loop for split online RL (Algorithm 1 rollout only).

Each step: get obs → send to GPU → receive action → execute on Franka → log transition.
Reward and episode termination come from RewardLogger (s/f keys or timeout).
Reset is handled by ResetManager before each episode (not part of RL step).

RUNTIME AUTHORITATIVE COPY: this file lives in ``smq&jgy/src`` and is placed FIRST on
the robot ``PYTHONPATH`` by ``smq&jgy/scripts/_env.sh``, so it overrides the
``rlt_project/rlt_reproduce/src`` copy at runtime. The Phase 4 transition logic
(``build_chunk_transitions`` / ``run_episode`` / ``_validate_chunk_transition``) is
kept byte-identical with the rlt_reproduce copy; only the robot-specific reset/camera
bootstrap and ``Path(__file__).parents[N]`` depths in ``main()`` differ (they must, the
two files sit at different directory depths). When editing the transition logic, apply
the SAME change to both copies.

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

from rlt.hardware.deoxys.collection_reset import collection_reset_settings, resolve_reset_yaml
from rlt.hardware.deoxys.camera_source import (
    build_deoxys_realsense_pair,
    flush_rgb_frame_cache,
    read_rgb_frames,
    wait_for_fresh_rgb_frames,
    wait_for_rgb_frames,
)
from rlt.hardware.deoxys.reset_manager import ResetManager
from rlt.hardware.gripper_factory import create_robot_env
from rlt.rl.gpu_client import GPUClient, MockGPUClient, create_gpu_client
from rlt.rl.reward_logger import EpisodeOutcome, RewardLogger
from rlt.rl.ws_protocol import ensure_observation_images_jpeg
from rlt.sim.mock_env import MockPrecisionEnv
from rlt.util.deoxys_paths import apply_deoxys_paths
from rlt.util.terminal_keys import flush_input, stdin_is_tty, terminal_keys

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


def _verify_reset_pose(
    env,
    reset_info: dict,
    *,
    max_pos_err_m: float,
) -> None:
    """Fail fast if reset did not reach init-cube height before VLA infer."""
    if not reset_info.get("workspace_reset") and not reset_info.get("external_reset"):
        return
    if reset_info.get("external_reset"):
        z = float(reset_info.get("ee_z_m", env.get_proprio()[2]))
        if z < 0.19:
            raise RuntimeError(
                f"External reset EE z={z:.4f}m too low (<0.19m) — aborting before VLA infer"
            )
        return

    err = float(reset_info.get("pos_err_m", 0.0))
    if err > max_pos_err_m:
        raise RuntimeError(
            f"Reset position error {err*100:.2f}cm exceeds limit {max_pos_err_m*100:.2f}cm — "
            "aborting episode before VLA infer"
        )
    if hasattr(env, "get_proprio") and reset_info.get("target_xyz"):
        proprio = env.get_proprio()
        target = np.asarray(reset_info["target_xyz"], dtype=np.float64)
        actual = proprio[:3].astype(np.float64)
        live_err = float(np.linalg.norm(actual - target))
        if live_err > max_pos_err_m:
            raise RuntimeError(
                f"Live EE error {live_err*100:.2f}cm vs target {target.round(4).tolist()} "
                f"(reset reported {err*100:.2f}cm) — wait longer or re-reset"
            )


def _post_reset_warmup(
    env,
    *,
    steps: int,
    control_hz: float,
    camera_manager,
    camera_mapping: dict[str, str],
    frame_cache: dict[str, np.ndarray] | None,
) -> None:
    """Hold at reset pose while cameras catch up (mirrors SFT post_reset_warmup_steps)."""
    if steps <= 0:
        return
    dt = 1.0 / control_hz
    for i in range(steps):
        if camera_manager is not None and camera_mapping:
            read_rgb_frames(camera_manager, camera_mapping, cache=frame_cache)
        if hasattr(env, "get_proprio"):
            env.get_proprio()
        if i == 0 or i == steps - 1:
            if hasattr(env, "get_proprio"):
                z = float(env.get_proprio()[2])
                console.print(f"[dim]post-reset warmup[/dim] {i + 1}/{steps} ee_z={z:.4f}m")
        time.sleep(dt)


def _build_observation(
    proprio: np.ndarray,
    camera_manager,
    camera_mapping: dict[str, str],
    language: str,
    *,
    frame_cache: dict[str, np.ndarray] | None = None,
    wait_timeout_sec: float = 3.0,
) -> dict[str, Any]:
    obs: dict[str, Any] = {
        "proprio": proprio.astype(np.float32),
        "language": language,
    }
    if camera_manager is None or not camera_mapping:
        return obs

    cache = frame_cache if frame_cache is not None else {}
    read_rgb_frames(camera_manager, camera_mapping, cache=cache, allow_stale=True)
    missing = [name for name in camera_mapping if name not in cache]
    if missing:
        try:
            wait_for_rgb_frames(
                camera_manager,
                camera_mapping,
                cache=cache,
                timeout_sec=wait_timeout_sec,
            )
        except TimeoutError as exc:
            raise RuntimeError(
                f"RealSense frames unavailable for {missing}. "
                "deoxys get_all_latest_frames() only returns new frames since last read — "
                "check cameras with: bash scripts/verify_camera_roles.sh"
            ) from exc

    images = {name: cache[name] for name in camera_mapping if name in cache}
    if len(images) == len(camera_mapping):
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


def build_chunk_transitions(
    states: list[np.ndarray],
    actions: list[np.ndarray],
    references: list[np.ndarray],
    rewards: list[float],
    *,
    chunk_length: int,
    stride: int,
    terminated: bool,
    interventions: list[bool] | None = None,
) -> list[dict[str, Any]]:
    """Assemble paper-faithful chunk transitions from an episode's per-step stream.

    Inputs are the continuous per-step episode stream:
      ``states``     = x_0 .. x_T           (len T+1, encoded RL states)
      ``actions``    = a_0 .. a_{T-1}        (executed actions, len T)
      ``references`` = ref_0 .. ref_{T-1}    (aligned VLA references, len T)
      ``rewards``    = r_0 .. r_{T-1}        (per-step rewards, len T)

    Emits, for each subsampled start ``s`` (stride, RLT App. B uses stride 2) with a
    full window ``s+C <= T``, the transition

        (x_s, a_{s:s+C}, ref_{s:s+C}, ref_{s+C:s+2C}, r_{s:s+C}, x_{s+C}, done)

    i.e. a REAL executed action chunk (no tiling), the aligned reference chunk, the
    NEXT-state reference chunk (for the critic target a'~pi(x',ref')), the per-step
    rewards (server discounts them into R = sum gamma^k r), and the state exactly C
    env-steps later. When the episode terminates, the terminal chunk (carrying the
    sparse success reward and ``done=True``) is always emitted, padded to length C if
    the episode ended in fewer than C steps.
    """
    C = int(chunk_length)
    stride = max(1, int(stride))
    T = len(actions)
    if not (len(states) == T + 1 and len(references) == T and len(rewards) == T):
        raise ValueError(
            f"stream length mismatch: states={len(states)} actions={T} "
            f"references={len(references)} rewards={len(rewards)} (need states == actions + 1)"
        )
    if T == 0:
        return []

    zero = np.zeros_like(np.asarray(references[0], dtype=np.float32))

    def chunk(seq: list[np.ndarray], start: int) -> np.ndarray:
        out = [np.asarray(x, dtype=np.float32) for x in seq[start : start + C]]
        if not out:
            out = [zero.copy()]
        while len(out) < C:
            out.append(out[-1].copy())
        return np.stack(out[:C]).astype(np.float32)

    def reward_chunk(start: int) -> list[float]:
        out = [float(r) for r in rewards[start : start + C]]
        while len(out) < C:
            out.append(0.0)
        return out[:C]

    def intervened_frac(start: int) -> float:
        if not interventions:
            return 0.0
        window = interventions[start : start + C]
        if not window:
            return 0.0
        return float(sum(1 for x in window if x)) / float(len(window))

    transitions: list[dict[str, Any]] = []
    emitted: set[int] = set()

    def emit(s: int, *, done: bool) -> None:
        e = s + C
        transitions.append(
            {
                "state": np.asarray(states[s], dtype=np.float32),
                "action": chunk(actions, s),
                "reference_action": chunk(references, s),
                "next_reference_action": chunk(references, e),
                "rewards": reward_chunk(s),
                "next_state": np.asarray(states[min(e, T)], dtype=np.float32),
                "done": bool(done),
                "intervened": intervened_frac(s),
                "_t_start": int(s),
                "_t_next": int(min(e, T)),
                "_n_real_steps": int(min(e, T) - s),
            }
        )
        emitted.add(s)

    for s in range(0, max(0, T - C + 1), stride):
        emit(s, done=bool(terminated and s + C == T))

    if terminated:
        s_term = max(0, T - C)
        if s_term not in emitted:
            emit(s_term, done=True)

    return transitions


def _validate_chunk_transition(tr: dict[str, Any], *, chunk_length: int, action_dim: int) -> None:
    """Fail loudly on any malformed chunk transition (Phase 4 debug validation)."""
    C, ad = int(chunk_length), int(action_dim)
    for key in ("action", "reference_action", "next_reference_action"):
        arr = np.asarray(tr[key], dtype=np.float32)
        if arr.shape != (C, ad):
            raise ValueError(f"transition {key} shape {arr.shape} != required ({C}, {ad})")
    if len(tr["rewards"]) != C:
        raise ValueError(f"transition rewards len {len(tr['rewards'])} != C={C}")
    gap = int(tr["_t_next"]) - int(tr["_t_start"])
    if not (gap == C or tr["done"]):
        raise ValueError(f"non-terminal transition temporal gap {gap} != C={C}")


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
    action_dim: int = 7,
    subsample_stride: int = 2,
    image_size: tuple[int, int] | None = None,
    frame_cache: dict[str, np.ndarray] | None = None,
    teleop=None,
) -> EpisodeOutcome:
    """One critical-phase episode: infer chunks, execute, build chunk transitions.

    Phase 4: instead of shipping one malformed single-step transition per env step
    (later tiled into a fake chunk on the GPU), the robot records the continuous
    per-step episode stream (states x_0..x_T via infer + per-step ``encode``, executed
    actions, aligned references, rewards) and, at episode end, assembles REAL chunk
    transitions ``(x_s, a_{s:s+C}, ref_{s:s+C}, R, x_{s+C})`` with a genuine C-step
    gap (see :func:`build_chunk_transitions`).
    """
    dt = 1.0 / control_hz
    proprio = env.get_proprio() if hasattr(env, "get_proprio") else env.reset()
    step = 0
    total_reward = 0.0
    records: list[StepRecord] = []
    episode_id = f"ep_{episode_index:04d}"

    # Continuous per-step episode stream used to assemble paper-faithful chunks.
    stream_states: list[np.ndarray] = []  # x_0 .. x_T
    stream_actions: list[np.ndarray] = []  # a_0 .. a_{T-1}
    stream_refs: list[np.ndarray] = []  # ref_0 .. ref_{T-1}
    stream_rewards: list[float] = []  # r_0 .. r_{T-1}
    stream_interventions: list[bool] = []  # human-takeover flag per step
    terminated = False
    outcome: EpisodeOutcome | None = None

    while not terminated:
        obs = _build_observation(
            proprio,
            camera_manager,
            camera_mapping,
            language,
            frame_cache=frame_cache,
        )
        if step == 0:
            preview = ensure_observation_images_jpeg(obs, image_size=image_size)
            images = preview.get("images") or obs.get("images") or {}
            jpegs = preview.get("images_jpeg") or {}
            ee_z = float(proprio[2])
            console.print(
                f"[dim]first infer obs[/dim] keys={sorted(obs.keys())} "
                f"images={sorted(images.keys())} "
                f"images_jpeg={sorted(jpegs.keys())} "
                f"jpeg_lens external={len(jpegs.get('external', ''))} "
                f"wrist={len(jpegs.get('wrist', ''))} "
                f"proprio_shape={np.asarray(proprio).shape} ee_z={ee_z:.4f}m"
            )
            if not jpegs:
                raise RuntimeError(
                    "first infer missing images_jpeg — RealSense returned no RGB frames "
                    f"(cache keys={sorted((frame_cache or {}).keys())})"
                )
            console.print(
                "[yellow]First GPU VLA infer may take 30–90s (CUDA warmup). Please wait…[/yellow]"
            )
        result = gpu.infer(obs)
        if step == 0:
            # Drop keys buffered during reset + the first CUDA-warmup infer so a
            # stray 'q'/'f'/'s' doesn't terminate the episode at step 1
            # (reason=quit, steps=1). See terminal_keys.flush_input.
            flush_input()
        if result.state is None:
            raise RuntimeError("GPU infer missing encoded state — restart GPU rl_server with latest code")
        # State at the current planning boundary (recomputed each infer; identical to
        # the last per-step encode, so no double counting in the stream).
        cur_state = np.asarray(result.state, dtype=np.float32)
        action_chunk = result.action_chunk
        ref_chunk = result.reference_action
        policy_mode = (result.meta or {}).get("policy_mode", "reference")
        exec_chunk = ref_chunk if policy_mode == "reference" else action_chunk
        if step == 0 and result.meta:
            console.print(f"[dim]GPU infer meta[/dim] {result.meta}")
            console.print(
                f"[dim]first action ({policy_mode})[/dim] pos={exec_chunk[0][:3].round(4).tolist()} "
                f"gripper_raw={float(exec_chunk[0][6]):.4f} (latched→1.0 if gripper_latch)"
            )
        n_exec = min(chunk_length, execute_prefix, len(exec_chunk))

        for i in range(n_exec):
            action = exec_chunk[i].astype(np.float32)
            ref_step = ref_chunk[i].astype(np.float32)

            # Human teleop takeover (RLT Sec. V), if enabled and engaged. The
            # teleop action is already in physical EE-delta units, executed via the
            # same env.step (Phase-1 safety clamp preserved); on intervened steps
            # the human action also REPLACES the stored reference (BC target).
            intervened = False
            if teleop is not None:
                teleop_action, engaged = teleop.poll()
                if engaged and teleop_action is not None:
                    action = teleop_action.astype(np.float32)
                    ref_step = teleop_action.astype(np.float32)
                    intervened = True

            ee_before = proprio[:3].copy()

            # Record the pre-step state x_step and the executed/reference action.
            stream_states.append(cur_state)
            stream_actions.append(action)
            stream_refs.append(ref_step)
            stream_interventions.append(intervened)

            next_proprio, _, _, _ = env.step(action)
            step += 1
            if step <= 5:
                delta_cm = (next_proprio[:3] - ee_before) * 100.0
                console.print(
                    f"[dim]step {step}[/dim] ee_delta_cm={delta_cm.round(2).tolist()} "
                    f"action_pos_m={action[:3].round(5).tolist()}"
                )

            outcome = reward_logger.poll(step)
            step_reward = float(outcome.reward) if outcome else 0.0
            step_done = bool(outcome.done) if outcome else False
            total_reward += step_reward
            stream_rewards.append(step_reward)

            records.append(
                StepRecord(
                    step=step,
                    proprio=proprio.tolist(),
                    action=action.tolist(),
                    reward=step_reward,
                    done=step_done,
                )
            )

            # Encode the resulting state x_{step+1} so the stream carries a genuine
            # per-step next_state (never a 1-step gap tiled to look like a chunk).
            next_obs = _build_observation(
                next_proprio,
                camera_manager,
                camera_mapping,
                language,
                frame_cache=frame_cache,
            )
            cur_state = np.asarray(gpu.encode(next_obs), dtype=np.float32)
            proprio = next_proprio

            if outcome:
                terminated = True
                stream_states.append(cur_state)  # terminal state x_T
                break

            time.sleep(dt)

    # --- Episode finished: assemble + ship REAL chunk transitions -----------------
    transitions = build_chunk_transitions(
        stream_states,
        stream_actions,
        stream_refs,
        stream_rewards,
        chunk_length=chunk_length,
        stride=subsample_stride,
        terminated=terminated,
        interventions=stream_interventions,
    )
    last_resp: dict[str, Any] = {}
    for tr in transitions:
        _validate_chunk_transition(tr, chunk_length=chunk_length, action_dim=action_dim)
        payload = {k: v for k, v in tr.items() if not k.startswith("_")}
        last_resp = gpu.send_transition(payload)
        if last_resp.get("updated") and last_resp.get("metrics"):
            console.print(
                f"[dim]learner[/dim] buffer={last_resp.get('buffer_size')} "
                f"metrics={last_resp.get('metrics')}"
            )
    console.print(
        f"[dim]sent {len(transitions)} chunk transitions[/dim] "
        f"stride={subsample_stride} C={chunk_length} steps={step} "
        f"buffer≈{last_resp.get('buffer_size', 'n/a')}"
    )

    if outcome is not None:
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
    raise RuntimeError("episode ended without an outcome")


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot-side actor loop for split online RL")
    parser.add_argument("--config", type=Path, default=Path("configs/plug_insertion.yaml"))
    parser.add_argument("--mock", action="store_true", help="Mock env + mock GPU (no robot)")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--gpu-host", default=None, help="Override gpu_server.host / GPU_SERVER_HOST")
    parser.add_argument("--reset-mode", default=None, choices=["demo", "demo_fast", "workspace", "home", "none"])
    parser.add_argument("--no-cameras", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    apply_deoxys_paths(raw, smq_root=Path(__file__).resolve().parents[4])

    rl = raw.get("online_rl", {})
    sc = raw.get("sft_collection", {})
    vla = raw.get("vla", {})
    paths = raw.get("paths", {})
    chunk_length = rl.get("chunk_length", 10)
    action_dim = rl.get("action_dim", 7)
    subsample_stride = int(rl.get("subsample_stride", 2))
    execute_prefix = vla.get("execute_prefix", 20)
    control_hz = rl.get("control_hz", raw.get("robot", {}).get("control_hz", 20))
    max_steps = args.max_steps or int(os.environ.get("MAX_STEPS", rl.get("max_steps_per_episode", 200)))
    episodes = args.episodes or int(os.environ.get("EPISODES", rl.get("max_episodes", 10)))
    language = vla.get("language_instruction", "")
    post_reset_settle_sec = float(
        rl.get("post_reset_settle_sec", sc.get("post_reset_settle_sec", 1.0))
    )
    post_reset_warmup_steps = int(
        rl.get("post_reset_warmup_steps", sc.get("post_reset_warmup_steps", 25))
    )
    post_reset_max_pos_err_m = float(
        rl.get(
            "post_reset_max_pos_err_m",
            collection_reset_settings(resolve_reset_yaml(raw, smq_root=Path(__file__).resolve().parents[3]))[1].pos_tol_m * 2.0,
        )
    )

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
    camera_frame_cache: dict[str, np.ndarray] = {}

    cam_cfg = raw.get("cameras", {})
    img_size_raw = cam_cfg.get("image_size")
    image_size = tuple(img_size_raw) if img_size_raw else None

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
            if camera_manager is not None:
                console.print("[cyan]Warming RealSense frame cache ...[/cyan]")
                wait_for_rgb_frames(
                    camera_manager,
                    camera_mapping,
                    cache=camera_frame_cache,
                    timeout_sec=15.0,
                )
                shapes = {k: v.shape for k, v in camera_frame_cache.items()}
                console.print(
                    f"[green]Cameras ready[/green] cache={sorted(camera_frame_cache.keys())} "
                    f"shapes={shapes}"
                )
        if isinstance(gpu, MockGPUClient):
            console.print("[green]Real robot env[/green] [dim](mock GPU — zero actions)[/dim]")
        else:
            console.print(f"[green]Real robot env[/green] [cyan]GPU ws://{gpu_host}[/cyan]")

    reset_manager = ResetManager.from_config(env, raw, rlt_root=rlt_root) if not env_mock else None
    if reset_manager is not None and args.reset_mode:
        from rlt.hardware.deoxys.reset_manager import ResetMode

        reset_manager.mode = ResetMode(args.reset_mode)
    if reset_manager is not None:
        console.print(
            f"[cyan]Reset mode[/cyan]: {reset_manager.mode.value} "
            f"(bash scripts/reset_to_init.sh — external subprocess per episode)"
        )

    teleop = None
    if not env_mock and hasattr(env, "controller_action_scales"):
        from rlt.teleop.intervention import build_intervention

        ctrl_trans, ctrl_rot = env.controller_action_scales()
        controller_type = raw.get("data_collection", {}).get("controller_type", "OSC_POSE")
        teleop = build_intervention(
            raw,
            ctrl_trans=ctrl_trans,
            ctrl_rot=ctrl_rot,
            controller_type=controller_type,
            action_dim=action_dim,
        )
        if teleop is not None:
            console.print(
                "[magenta]Teleop intervention ENABLED[/magenta] — push the SpaceMouse to take "
                "over; release to hand control back to the policy (s/f/q unchanged)"
            )

    console.print(
        "Controls: [green]s[/green]=success (reward=1)  "
        "[red]f[/red]=fail  [yellow]q[/yellow]=quit  "
        f"timeout={max_steps} steps"
    )

    key_ctx = nullcontext() if (env_mock or not stdin_is_tty()) else terminal_keys()
    prev_success = False
    try:
        with key_ctx:
            for ep in range(episodes):
                reset_info: dict = {}
                use_external_reset = (
                    reset_manager is not None
                    and reset_manager.mode.value == "workspace"
                    and reset_manager.reset_method == "external"
                )
                if use_external_reset and camera_manager is not None:
                    console.print(
                        "[cyan]Stopping cameras before external reset "
                        "(camera child processes inherit ZMQ port 5555)[/cyan]"
                    )
                    camera_manager.stop()
                    camera_manager = None
                    flush_rgb_frame_cache(camera_frame_cache)

                if reset_manager is not None:
                    proprio, reset_info = reset_manager.reset(prev_success=prev_success)
                    console.print(f"Reset ep {ep}: {reset_info}")
                    if not env_mock:
                        _verify_reset_pose(
                            env,
                            reset_info,
                            max_pos_err_m=post_reset_max_pos_err_m,
                        )
                elif hasattr(env, "reset"):
                    env.reset()

                if use_external_reset and not args.no_cameras and cam_cfg.get("mapping"):
                    console.print("[cyan]Restarting cameras after external reset ...[/cyan]")
                    camera_manager, camera_mapping = _setup_cameras(raw, raw["robot"])
                    wait_for_rgb_frames(
                        camera_manager,
                        camera_mapping,
                        cache=camera_frame_cache,
                        timeout_sec=15.0,
                    )

                if camera_manager is not None:
                    console.print(
                        f"[cyan]Post-reset settle[/cyan] {post_reset_settle_sec:.1f}s + "
                        f"fresh cameras (clearing stale cache)"
                    )
                    wait_for_fresh_rgb_frames(
                        camera_manager,
                        camera_mapping,
                        cache=camera_frame_cache,
                        timeout_sec=10.0,
                        settle_sec=post_reset_settle_sec,
                    )
                    _post_reset_warmup(
                        env,
                        steps=post_reset_warmup_steps,
                        control_hz=control_hz,
                        camera_manager=camera_manager,
                        camera_mapping=camera_mapping,
                        frame_cache=camera_frame_cache,
                    )
                    if reset_info.get("target_xyz") and hasattr(env, "get_proprio"):
                        target_z = float(reset_info["target_xyz"][2])
                        live_z = float(env.get_proprio()[2])
                        console.print(
                            f"[green]Ready for VLA[/green] target_z={target_z:.4f}m "
                            f"live_z={live_z:.4f}m Δz={(live_z - target_z)*100:.2f}cm"
                        )

                outcome = run_episode(
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
                    action_dim=action_dim,
                    subsample_stride=subsample_stride,
                    image_size=image_size,
                    frame_cache=camera_frame_cache,
                    teleop=teleop,
                )
                prev_success = bool(outcome is not None and outcome.reason == "success_key")
                if outcome is not None and outcome.reason == "quit":
                    console.print("[yellow]q pressed → stopping rollout run[/yellow]")
                    break
    finally:
        if teleop is not None:
            teleop.close()
        if camera_manager is not None:
            camera_manager.stop()
        if hasattr(env, "close"):
            env.close()
        gpu.close()


if __name__ == "__main__":
    main()
