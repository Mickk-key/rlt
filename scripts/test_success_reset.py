#!/usr/bin/env python3
"""Standalone real-robot test for the post-success reset path (reference mode).

Manually place the robot in the SUCCESS state first: plug inserted, gripper still
holding it. This script then calls the SAME production code as the actor loop's
per-episode reset after a success:

    reset_manager.reset(prev_success=True)

It runs ONLY:
  1. vertical z-lift with x/y fixed and gripper closed (the success-lift branch);
  2. the normal workspace / external归位 reset;
  3. exit — no new episode, no policy inference, no rollout, no training,
     no replay-buffer changes, no retries, no checkpoint writes.

It prints EE position before lift, after lift, and after reset (plus x/y/z
displacement), the imported production file paths (duplicate code trees exist),
and requires CONFIRM=1 before any robot motion.

Usage:
  python scripts/test_success_reset.py --dry-run
  CONFIRM=1 python scripts/test_success_reset.py --config configs/plug_insertion.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _setup_pythonpath() -> tuple[Path, Path]:
    """Mirror scripts/_env.sh: smq/src overrides rlt_reproduce/src on sys.path."""
    smq_root = Path(os.environ.get("SMQ_ROOT") or Path(__file__).resolve().parents[1]).resolve()
    rlt_root = Path(
        os.environ.get("RLT_ROOT") or (smq_root / "rlt_project" / "rlt_reproduce")
    ).resolve()
    # Insert rlt_reproduce/src first, then smq/src, so smq/src has highest priority
    # (final order == PYTHONPATH "smq/src:rlt_reproduce/src").
    for p in (rlt_root / "src", smq_root / "src"):
        sp = str(p)
        if sp in sys.path:
            sys.path.remove(sp)
        sys.path.insert(0, sp)
    return smq_root, rlt_root


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the post-success reset path (z-lift then归位) on the real robot.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/plug_insertion.yaml"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No robot connection and no motion — only verify imports / config resolution.",
    )
    parser.add_argument(
        "--lift-z",
        type=float,
        default=None,
        help="Override online_rl.success_lift_z_m (meters) for this run only.",
    )
    args = parser.parse_args()

    smq_root, rlt_root = _setup_pythonpath()

    import numpy as np
    import yaml

    # Import production modules and print their resolved paths (duplicate trees exist).
    from rlt.hardware import gripper_factory
    from rlt.hardware.deoxys import collection_reset as collection_reset_mod
    from rlt.hardware.deoxys import deoxys_env as deoxys_env_mod
    from rlt.hardware.deoxys import fast_reset as fast_reset_mod
    from rlt.hardware.deoxys import reset_manager as reset_manager_mod
    from rlt.hardware.deoxys.reset_manager import ResetManager
    from rlt.hardware.gripper_factory import create_robot_env
    from rlt.util import deoxys_paths
    from rlt.util.deoxys_paths import apply_deoxys_paths

    print("=" * 78)
    print("Imported production modules (verify which duplicate code tree is running):")
    for name, mod in [
        ("reset_manager", reset_manager_mod),
        ("fast_reset", fast_reset_mod),
        ("deoxys_env", deoxys_env_mod),
        ("collection_reset", collection_reset_mod),
        ("gripper_factory", gripper_factory),
        ("deoxys_paths", deoxys_paths),
    ]:
        print(f"  {name:16s} -> {getattr(mod, '__file__', '?')}")
    print(
        f"  {'lift_ee_z':16s} -> "
        f"{fast_reset_mod.lift_ee_z.__module__} @ "
        f"{Path(fast_reset_mod.__file__).name}"
    )
    print(f"  {'ResetManager':16s} -> defined in {ResetManager.__module__}")
    print(f"  SMQ_ROOT = {smq_root}")
    print(f"  RLT_ROOT = {rlt_root}")
    print("=" * 78)

    cfg_path = args.config if args.config.is_absolute() else (smq_root / args.config)
    if not cfg_path.is_file():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    apply_deoxys_paths(raw, smq_root=smq_root)
    if args.lift_z is not None:
        raw.setdefault("online_rl", {})["success_lift_z_m"] = float(args.lift_z)

    rl = raw.get("online_rl", {})
    robot = raw.get("robot", {})
    print(f"config: {cfg_path}")
    print(f"  reset_mode        = {rl.get('reset_mode')}")
    print(f"  reset_method      = {rl.get('reset_method')}")
    print(f"  reset_config      = {rl.get('reset_config')}")
    print(f"  success_lift_z_m  = {rl.get('success_lift_z_m')}")
    print(f"  deoxys_root       = {robot.get('deoxys_root')}")
    print(f"  deoxys_config     = {robot.get('deoxys_config')}")

    if args.dry_run:
        print("\n[DRY-RUN] No robot connection, no motion. Imports + config resolved. Exiting.")
        return 0

    if os.environ.get("CONFIRM") != "1":
        print(
            "\nERROR: real-robot motion requires CONFIRM=1.\n"
            "  Place the robot in the success state (plug inserted, gripper holding),\n"
            "  clear the workspace, keep the e-stop ready, then run:\n"
            f"    CONFIRM=1 python scripts/test_success_reset.py --config {args.config}\n"
            "  Or verify without motion:\n"
            "    python scripts/test_success_reset.py --dry-run",
            file=sys.stderr,
        )
        return 1

    env = None
    try:
        # Reproduce the production cwd (run_actor_loop.sh does `cd "$RLT_ROOT"`), so
        # relative config paths like gripper.config=configs/franka/franka_hand.yaml
        # resolve under rlt_reproduce.
        os.chdir(rlt_root)
        print(f"\ncwd set to RLT_ROOT for relative config resolution: {rlt_root}")
        print("Creating real robot env (Franka FR3 via deoxys) ...")
        env = create_robot_env(raw, rlt_root=rlt_root)
        reset_manager = ResetManager.from_config(env, raw, rlt_root=rlt_root)
        print(
            f"ResetManager: mode={reset_manager.mode.value} "
            f"method={reset_manager.reset_method} "
            f"success_lift_z_m={reset_manager.success_lift_z_m}"
        )

        before = np.asarray(env.get_proprio()[:3], dtype=float)
        print(f"\n[EE] before lift  : {before.round(4).tolist()}")

        print("\n>>> Calling production reset_manager.reset(prev_success=True) <<<\n")
        proprio, reset_info = reset_manager.reset(prev_success=True)

        lift = reset_manager.last_lift_info or {}
        after_lift = np.asarray(lift.get("after_xyz", before.tolist()), dtype=float)
        after_reset = np.asarray(proprio[:3], dtype=float)

        print("\n" + "=" * 78)
        print("RESULT")
        print(f"[EE] before lift  : {before.round(4).tolist()}")
        print(
            f"[EE] after  lift  : {after_lift.round(4).tolist()}  "
            f"(success-lift applied={lift.get('applied')}, lift_z_m={lift.get('lift_z_m')})"
        )
        print(f"[EE] after  reset : {after_reset.round(4).tolist()}")
        print(f"  Δ lift   (cm) xyz = {((after_lift - before) * 100).round(2).tolist()}")
        print(f"  Δ reset  (cm) xyz = {((after_reset - after_lift) * 100).round(2).tolist()}")
        print(f"  Δ total  (cm) xyz = {((after_reset - before) * 100).round(2).tolist()}")
        print(f"  reset_info = {reset_info}")
        print("=" * 78)
        print("\nDone. Not starting another episode.")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user — shutting down safely.", file=sys.stderr)
        return 130
    except Exception as exc:  # shut down safely on any failure
        import traceback

        print(f"\nERROR during reset test: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        if env is not None and hasattr(env, "close"):
            try:
                env.close()
                print("[cleanup] env closed.")
            except Exception as exc:  # best-effort cleanup
                print(f"[cleanup] env.close() failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
