#!/usr/bin/env python3
"""Standalone demo-driven reset stability test on the robot PC.

Runs N consecutive demo resets, logs per-trial errors/timing, and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from rlt.hardware.deoxys.demo_reset import (
    DemoResetSampler,
    ResetPose,
    load_reset_poses,
    move_to_demo_pose,
    pose_errors,
)
from rlt.hardware.deoxys.deoxys_env import DeoxysEnv
from rlt.util.deoxys_paths import resolve_demo_reset_path


@dataclass
class TrialResult:
    trial: int
    success: bool
    episode_id: str
    demo_index: int
    duration_sec: float
    steps: int | None
    pos_err_cm: float | None
    rot_err_deg: float | None
    gripper_err_mm: float | None
    error: str | None = None


def _ensure_demo_json(demo_path: Path, episodes_dir: Path) -> None:
    json_files = list(demo_path.glob("*.json"))
    if json_files:
        return
    npz_files = list(episodes_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"No demo JSON in {demo_path} and no NPZ in {episodes_dir}. "
            "Run: PYTHONPATH=src python -m rlt.scripts.export_demo_reset_json"
        )
    from rlt.scripts.export_demo_reset_json import export_npz_dir

    demo_path.mkdir(parents=True, exist_ok=True)
    n = export_npz_dir(episodes_dir, demo_path)
    print(f"[prep] exported {n} JSON initial states -> {demo_path}")


def _current_errors(env: DeoxysEnv, target: ResetPose) -> tuple[float, float, float]:
    proprio = env.get_proprio()
    pos = proprio[:3]
    quat = proprio[3:7]
    pos_err, rot_err = pose_errors(pos, quat, target)
    grip_err = abs(float(proprio[7]) - target.gripper_width)
    return pos_err * 100.0, rot_err, grip_err * 1000.0


def run_trials_env(
    env: DeoxysEnv,
    *,
    trials: int,
    pause_sec: float,
    fast: bool = False,
) -> list[TrialResult]:
    results: list[TrialResult] = []
    for i in range(trials):
        t0 = time.time()
        try:
            proprio = env.reset(fast=fast)
            info = env.last_reset_info
            if not info.get("demo_reset"):
                raise RuntimeError("use_demo_reset is false — enable it in config")

            target = ResetPose(
                ee_pose=proprio[:3],
                quaternion=proprio[3:7],
                gripper_width=float(proprio[7]),
                episode_id=str(info.get("episode_id", "?")),
            )
            pos_cm, rot_deg, grip_mm = _current_errors(env, target)
            dur = time.time() - t0
            results.append(
                TrialResult(
                    trial=i + 1,
                    success=True,
                    episode_id=str(info.get("episode_id", "?")),
                    demo_index=int(info.get("demo_index", -1)),
                    duration_sec=dur,
                    steps=int(info.get("steps", 0)),
                    pos_err_cm=pos_cm,
                    rot_err_deg=rot_deg,
                    gripper_err_mm=grip_mm,
                )
            )
            print(
                f"[{i+1}/{trials}] OK {info.get('episode_id')} "
                f"idx={info.get('demo_index')} "
                f"pos={pos_cm:.2f}cm rot={rot_deg:.2f}deg grip={grip_mm:.1f}mm "
                f"time={dur:.1f}s steps={info.get('steps')}"
            )
        except Exception as exc:
            dur = time.time() - t0
            results.append(
                TrialResult(
                    trial=i + 1,
                    success=False,
                    episode_id="",
                    demo_index=-1,
                    duration_sec=dur,
                    steps=None,
                    pos_err_cm=None,
                    rot_err_deg=None,
                    gripper_err_mm=None,
                    error=str(exc),
                )
            )
            print(f"[{i+1}/{trials}] FAIL {exc}")

        if pause_sec > 0 and i + 1 < trials:
            time.sleep(pause_sec)
    return results


def run_trials_direct(env: DeoxysEnv, sampler: DemoResetSampler, *, trials: int, pause_sec: float) -> list[TrialResult]:
    """Call move_to_demo_pose directly (bypass env.reset wrapper)."""
    results: list[TrialResult] = []
    iface = env._iface
    assert iface is not None

    for i in range(trials):
        sample = sampler.sample_reset_pose()
        target = ResetPose(
            ee_pose=np.asarray(sample["ee_pose"], dtype=np.float32),
            quaternion=np.asarray(sample["quaternion"], dtype=np.float32),
            gripper_width=float(sample["gripper_width"]),
            episode_id=str(sample["episode_id"]),
            success=sample.get("success"),
        )
        t0 = time.time()
        try:
            info = move_to_demo_pose(
                iface,
                target,
                controller_cfg=env._controller_cfg,
                gripper=env._gripper,
                pos_tol_m=env.cfg.demo_reset_pos_tol_m,
                rot_tol_deg=env.cfg.demo_reset_rot_tol_deg,
                control_hz=env.cfg.control_hz,
                safety=env._demo_safety,
                joint_controller_cfg=env._joint_controller_cfg,
                osc_position_cfg=env._osc_position_cfg,
            )
            pos_cm, rot_deg, grip_mm = _current_errors(env, target)
            dur = time.time() - t0
            results.append(
                TrialResult(
                    trial=i + 1,
                    success=True,
                    episode_id=target.episode_id,
                    demo_index=int(sample["demo_index"]),
                    duration_sec=dur,
                    steps=int(info.get("steps", 0)),
                    pos_err_cm=pos_cm,
                    rot_err_deg=rot_deg,
                    gripper_err_mm=grip_mm,
                )
            )
            print(
                f"[{i+1}/{trials}] OK {target.episode_id} idx={sample['demo_index']} "
                f"pos={pos_cm:.2f}cm rot={rot_deg:.2f}deg grip={grip_mm:.1f}mm time={dur:.1f}s"
            )
        except Exception as exc:
            dur = time.time() - t0
            results.append(
                TrialResult(
                    trial=i + 1,
                    success=False,
                    episode_id=target.episode_id,
                    demo_index=int(sample["demo_index"]),
                    duration_sec=dur,
                    steps=None,
                    pos_err_cm=None,
                    rot_err_deg=None,
                    gripper_err_mm=None,
                    error=str(exc),
                )
            )
            print(f"[{i+1}/{trials}] FAIL {target.episode_id} idx={sample['demo_index']}: {exc}")

        if pause_sec > 0 and i + 1 < trials:
            time.sleep(pause_sec)
    return results


def print_summary(results: list[TrialResult], *, pos_limit_cm: float, rot_limit_deg: float) -> int:
    ok = [r for r in results if r.success]
    fail = [r for r in results if not r.success]
    within = [
        r
        for r in ok
        if r.pos_err_cm is not None and r.pos_err_cm <= pos_limit_cm
    ]

    print("\n========== demo reset stability summary ==========")
    print(f"  trials:        {len(results)}")
    print(f"  success:       {len(ok)}")
    print(f"  failed:        {len(fail)}")
    print(f"  within tol:    {len(within)} (pos<={pos_limit_cm}cm)")

    if ok:
        pos = [r.pos_err_cm for r in ok if r.pos_err_cm is not None]
        rot = [r.rot_err_deg for r in ok if r.rot_err_deg is not None]
        dur = [r.duration_sec for r in ok]
        print(f"  pos err (cm):  mean={np.mean(pos):.2f} max={np.max(pos):.2f}")
        print(f"  rot err (deg): mean={np.mean(rot):.2f} max={np.max(rot):.2f}")
        print(f"  duration (s):  mean={np.mean(dur):.1f} max={np.max(dur):.1f}")

    if fail:
        print("  failures:")
        for r in fail:
            print(f"    trial {r.trial}: {r.error}")

    return 0 if len(fail) == 0 and len(within) == len(ok) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo-driven reset stability test")
    parser.add_argument("--config", type=Path, default=Path("configs/plug_insertion.yaml"))
    parser.add_argument("--trials", type=int, default=10, help="Number of consecutive resets")
    parser.add_argument("--pause-sec", type=float, default=1.0, help="Pause between trials")
    parser.add_argument("--demo-path", type=str, default=None, help="Override demo_reset_path")
    parser.add_argument(
        "--mode",
        choices=["env", "direct"],
        default="env",
        help="env=DeoxysEnv.reset(); direct=move_to_demo_pose only",
    )
    parser.add_argument(
        "--reset-mode",
        choices=["demo", "demo_fast"],
        default="demo",
        help="demo=full home+demo; demo_fast=skip home when safe (online RL)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate demo JSON/NPZ loading; no robot connection",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Save per-trial JSON report (default: logs/demo_reset_report.json under smq root)",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    smq_root = config_path.parents[1]
    rlt_root = smq_root / "rlt_project" / "rlt_reproduce"
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    demo_path = resolve_demo_reset_path(raw, smq_root=smq_root, override=args.demo_path)
    episodes_dir = (smq_root / raw["paths"]["episodes_dir"]).resolve()
    _ensure_demo_json(demo_path, episodes_dir)

    poses = load_reset_poses(demo_path)
    print(f"[prep] demo pool: {len(poses)} poses from {demo_path}")

    if args.dry_run:
        sampler = DemoResetSampler(demo_path, seed=0)
        for i in range(min(3, len(sampler))):
            p = sampler.sample_reset_pose()
            print(f"  sample {i}: {p['episode_id']} ee={np.round(p['ee_pose'], 3)}")
        print("[dry-run] OK — data load only, no robot motion")
        return 0

    raw = dict(raw)
    raw.setdefault("data_collection", {})
    raw.setdefault("paths", {})
    raw["data_collection"]["use_demo_reset"] = True
    raw["paths"]["demo_reset_path"] = str(
        demo_path.relative_to(smq_root) if demo_path.is_relative_to(smq_root) else demo_path
    )

    env = DeoxysEnv.from_rlt_config(raw, rlt_root=rlt_root)
    if env.cfg.backend != "deoxys":
        print("ERROR: robot.backend must be deoxys for live test", file=sys.stderr)
        return 1

    dc = raw["data_collection"]
    pos_lim = float(dc.get("demo_reset_pos_tol_m", 0.01)) * 100.0
    rot_lim = float(dc.get("demo_reset_rot_tol_deg", 5.0))

    fast = args.reset_mode == "demo_fast"
    print(
        f"[run] {args.trials} trials mode={args.mode} reset={args.reset_mode} "
        f"pause={args.pause_sec}s"
    )
    print(f"[run] thresholds: pos<={pos_lim:.1f}cm (rot logged, not required unless trim_orientation)")

    try:
        sampler = DemoResetSampler(demo_path)
        if args.mode == "env":
            results = run_trials_env(
                env,
                trials=args.trials,
                pause_sec=args.pause_sec,
                fast=fast,
            )
        else:
            results = run_trials_direct(env, sampler, trials=args.trials, pause_sec=args.pause_sec)
    finally:
        env.close()

    smq_root = rlt_root.parents[1]
    report_path = args.report
    if report_path is None:
        log_dir = smq_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        report_path = log_dir / "demo_reset_report.json"
    report_path = report_path.resolve()
    report_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[report] saved -> {report_path}")

    return print_summary(results, pos_limit_cm=pos_lim, rot_limit_deg=rot_lim)


if __name__ == "__main__":
    raise SystemExit(main())
