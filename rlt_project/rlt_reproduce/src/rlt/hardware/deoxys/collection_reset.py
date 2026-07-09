"""Init-cube reset shared by reset_to_init.sh, SFT collect, and online RL actor.

Uses the same ``reset_to_collection_init`` entry point as
``python -m rlt.scripts.reset_to_init_pose``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from rlt.hardware.deoxys.fast_reset import (
    FastResetConfig,
    FastResetResult,
    InitCubeConfig,
    reset_to_collection_init,
)
from rlt.teleop.spacemouse_control import DEFAULT_RESET_JOINTS


def resolve_reset_yaml(raw: dict, *, smq_root: Path | None = None) -> dict:
    """Merge optional ``online_rl.reset_config`` (e.g. sft_plug_insertion.yaml)."""
    rl = raw.get("online_rl", {})
    rel = rl.get("reset_config")
    if not rel:
        return raw
    path = Path(rel)
    if smq_root is not None and not path.is_absolute():
        path = (smq_root / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"online_rl.reset_config not found: {path}")
    with open(path, encoding="utf-8") as f:
        reset_raw = yaml.safe_load(f) or {}
    merged = dict(raw)
    merged["sft_collection"] = {
        **reset_raw.get("sft_collection", {}),
        **raw.get("sft_collection", {}),
    }
    merged["data_collection"] = {
        **reset_raw.get("data_collection", {}),
        **raw.get("data_collection", {}),
    }
    return merged


def collection_reset_settings(raw: dict) -> tuple[InitCubeConfig, FastResetConfig]:
    """Build InitCube + FastResetConfig exactly like ``reset_to_init_pose.py``."""
    sc = raw.get("sft_collection", {})
    dc = raw.get("data_collection", {})
    ws = InitCubeConfig.from_yaml_dict(sc.get("workspace_randomization", {}))
    home_joints = list(
        sc.get("reset_joint_positions", dc.get("reset_joint_positions")) or DEFAULT_RESET_JOINTS
    )
    fast = FastResetConfig(
        control_hz=float(sc.get("fps", 50.0)),
        pos_tol_m=float(sc.get("reset_pos_tol_m", 0.015)),
        home_joints=home_joints,
        joint_home_if_delta_above_m=float(sc.get("joint_home_if_delta_above_m", 0.35)),
        approach_xy_first=bool(sc.get("approach_xy_first", True)),
    )
    return ws, fast


def reset_to_init_cube(
    robot_interface,
    *,
    gripper,
    osc_position_cfg,
    joint_controller_cfg,
    raw: dict,
    randomize: bool = True,
    logger=None,
    max_attempts: int = 2,
) -> FastResetResult:
    """Run collection init reset with optional retry (same motion as reset_to_init.sh)."""
    ws_cfg, fast_cfg = collection_reset_settings(raw)
    sc = raw.get("sft_collection", {})
    max_attempts = int(sc.get("reset_max_attempts", max_attempts))
    last: FastResetResult | None = None
    if logger is None:
        log = print
    elif callable(getattr(logger, "info", None)):
        log = logger.info
    elif callable(getattr(logger, "print", None)):
        log = logger.print
    else:
        log = print

    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            log(f"[collection_reset] retry {attempt}/{max_attempts} (prev err too large)")
        last = reset_to_collection_init(
            robot_interface,
            gripper=gripper,
            cube_cfg=ws_cfg,
            osc_position_cfg=osc_position_cfg,
            joint_controller_cfg=joint_controller_cfg,
            reset_cfg=fast_cfg,
            randomize=randomize,
            logger=logger,
        )
        if last.pos_err_m <= fast_cfg.pos_tol_m * 2.0:
            return last
    assert last is not None
    return last


def wait_until_init_pose(
    get_proprio: Callable[[], np.ndarray],
    raw: dict,
    *,
    timeout_sec: float = 30.0,
    stable_reads: int = 8,
) -> np.ndarray:
    """Block until EE is at init-cube height/xy (prevents VLA infer while still low)."""
    ws_cfg, fast_cfg = collection_reset_settings(raw)
    floor_z = ws_cfg.floor_z
    min_z = floor_z - fast_cfg.pos_tol_m
    bc = np.asarray(ws_cfg.bottom_center_xyz, dtype=np.float64)
    half = ws_cfg.xy_half_range_m + 0.02

    stable = 0
    deadline = time.time() + timeout_sec
    last_pos: np.ndarray | None = None
    while time.time() < deadline:
        proprio = np.asarray(get_proprio(), dtype=np.float64)
        pos = proprio[:3]
        last_pos = pos
        z_ok = float(pos[2]) >= min_z
        xy_ok = bool(np.all(np.abs(pos[:2] - bc[:2]) <= half))
        if z_ok and xy_ok:
            stable += 1
            if stable >= stable_reads:
                return proprio.astype(np.float32)
        else:
            stable = 0
        time.sleep(0.1)

    pos_s = "?" if last_pos is None else last_pos.round(4).tolist()
    raise RuntimeError(
        f"Arm not at init cube after reset (waited {timeout_sec:.0f}s): "
        f"last_pos={pos_s} need z>={min_z:.4f} xy within {half*100:.0f}cm of {bc[:2].round(4).tolist()}"
    )


def resolve_rlt_reproduce_root(smq_root: Path, rlt_root: Path | None = None) -> Path:
    """``rlt_project/rlt_reproduce`` — same cwd as ``scripts/reset_to_init.sh``."""
    if rlt_root is not None:
        candidate = Path(rlt_root).resolve()
        if (candidate / "src" / "rlt" / "scripts" / "reset_to_init_pose.py").is_file():
            return candidate
        nested = candidate / "rlt_project" / "rlt_reproduce"
        if (nested / "src" / "rlt").is_dir():
            return nested.resolve()
    env_root = os.environ.get("RLT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return (smq_root / "rlt_project" / "rlt_reproduce").resolve()


def wait_for_deoxys_port_free(*, port: int = 5555, timeout_sec: float = 15.0) -> None:
    """Wait until the FrankaInterface ZMQ client port can be bound (post-close)."""
    import socket

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("0.0.0.0", port))
            return
        except OSError:
            time.sleep(0.25)
    raise RuntimeError(
        f"Deoxys ZMQ port {port} still in use after {timeout_sec:.0f}s. "
        "Stop camera processes and close FrankaInterface before external reset."
    )


def run_external_reset_subprocess(
    *,
    smq_root: Path,
    rlt_root: Path,
    config_path: Path,
    randomize: bool = True,
    logger: Any = None,
    skip_free_script: bool = True,
) -> None:
    """Identical to ``bash scripts/reset_to_init.sh`` (dedicated reset_to_init_pose client)."""
    log = print if logger is None else getattr(logger, "print", print)

    if not skip_free_script:
        free_script = smq_root / "scripts" / "free_deoxys_client.sh"
        if free_script.is_file():
            log("[external_reset] release ZMQ client (free_deoxys_client.sh) ...")
            subprocess.run(["bash", str(free_script)], cwd=str(smq_root), check=False)
    else:
        log("[external_reset] waiting for ZMQ port 5555 to be free ...")
        wait_for_deoxys_port_free(timeout_sec=15.0)

    cmd = [
        sys.executable,
        "-m",
        "rlt.scripts.reset_to_init_pose",
        "--config",
        str(config_path),
    ]
    if randomize:
        cmd.append("--random")

    rlt_reproduce = resolve_rlt_reproduce_root(smq_root, rlt_root)
    env = os.environ.copy()
    smq_src = smq_root / "src"
    rlt_src = rlt_reproduce / "src"
    env["PYTHONPATH"] = f"{smq_src}:{rlt_src}:{env.get('PYTHONPATH', '')}"
    env["SMQ_ROOT"] = str(smq_root)
    env["RLT_ROOT"] = str(rlt_reproduce)

    log(f"[external_reset] subprocess (cwd={rlt_reproduce}): {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(rlt_reproduce), check=True, env=env)
    log("[external_reset] subprocess finished — arm should be at init cube")

