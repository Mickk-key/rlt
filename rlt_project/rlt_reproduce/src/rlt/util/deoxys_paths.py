"""Resolve deoxys install path under SMQ&JGY (vendored copy preferred)."""

from __future__ import annotations

import os
from pathlib import Path


def smq_root_from_file(file_path: Path) -> Path:
    """SMQ root from ``smq&jgy/src/rlt/scripts/actor_loop.py`` (parents[3])."""
    return file_path.resolve().parents[3]


def smq_root_from_rlt(rlt_root: Path | None = None) -> Path:
    """Return SMQ&JGY repo root (``.../smq&jgy``)."""
    env = os.environ.get("SMQ_ROOT")
    if env:
        return Path(env).resolve()
    root = Path(rlt_root).resolve() if rlt_root is not None else Path(__file__).resolve().parents[4]
    if root.name == "rlt_reproduce":
        return root.parent.parent
    if root.name == "rlt_project":
        return root.parent
    if root.name == "src":
        return root.parent.parent
    return root


def resolve_deoxys_paths(
    *,
    smq_root: Path | None = None,
    interface_name: str = "charmander.yml",
    robot_cfg: dict | None = None,
) -> tuple[str, str]:
    """Return (deoxys_root, deoxys_config). Prefer ``smq&jgy/third_party/deoxys``."""
    smq = smq_root or smq_root_from_rlt(smq_root)
    vendored = smq / "third_party" / "deoxys"
    env_root = os.environ.get("DEOXYS_ROOT")
    robot_pc = Path("/home/host5010/workspaces/smq&jgy/third_party/deoxys")
    legacy = Path("/home/host5010/workspaces/wty/deoxys_control/deoxys")

    yaml_root = None
    if robot_cfg:
        raw_root = robot_cfg.get("deoxys_root")
        if raw_root:
            p = Path(raw_root)
            yaml_root = (smq / p) if not p.is_absolute() else p

    for candidate in (
        Path(env_root) if env_root else None,
        yaml_root,
        vendored,
        robot_pc,
        legacy,
    ):
        if candidate is None:
            continue
        cfg = candidate / "config" / interface_name
        if candidate.is_dir() and cfg.is_file():
            return str(candidate.resolve()), str(cfg.resolve())

    raise FileNotFoundError(
        f"deoxys not found under {vendored} (or via DEOXYS_ROOT / robot PC paths). "
        f"On robot PC run: bash {smq}/scripts/robot/copy_deoxys_to_smq.sh "
        f"or set DEOXYS_ROOT to a valid deoxys tree."
    )


def apply_deoxys_paths(raw: dict, *, smq_root: Path | None = None) -> dict:
    """Patch ``robot.deoxys_root`` / ``robot.deoxys_config`` in a loaded yaml dict."""
    robot = raw.setdefault("robot", {})
    root, cfg = resolve_deoxys_paths(smq_root=smq_root, robot_cfg=robot)
    robot["deoxys_root"] = root
    robot["deoxys_config"] = cfg
    return raw


def resolve_demo_reset_path(raw: dict, *, smq_root: Path, override: str | None = None) -> Path:
    """Resolve demo reset JSON/NPZ pool (``configs/plug_insertion.yaml``)."""
    dc = raw.get("data_collection", {})
    paths_cfg = raw.get("paths", {})
    rel = (
        override
        or paths_cfg.get("demo_reset_path")
        or dc.get("demo_reset_path")
        or paths_cfg.get("episodes_dir", "")
    )
    if not rel:
        raise ValueError("demo_reset_path not set in config")

    path = Path(rel)
    if path.is_absolute():
        return path.resolve()

    rlt_root = smq_root / "rlt_project" / "rlt_reproduce"
    for base in (smq_root, rlt_root):
        candidate = (base / path).resolve()
        if candidate.is_dir() and (
            list(candidate.glob("*.json")) or list(candidate.glob("*.npz"))
        ):
            return candidate
    return (smq_root / path).resolve()


def resolve_controller_cfg_path(
    cfg_name: str,
    *,
    smq_root: Path | None = None,
    deoxys_config_root: str | Path | None = None,
) -> Path:
    """Resolve OSC / joint controller yaml (SMQ ``configs/deoxys/`` preferred)."""
    smq = smq_root or smq_root_from_rlt()
    rel = Path(cfg_name)
    if rel.is_file():
        return rel.resolve()

    smq_candidate = smq / rel
    if smq_candidate.is_file():
        return smq_candidate.resolve()

    if deoxys_config_root is not None:
        deoxys_candidate = Path(deoxys_config_root) / rel.name
        if deoxys_candidate.is_file():
            return deoxys_candidate.resolve()

    raise FileNotFoundError(
        f"Controller config not found: {cfg_name!r} "
        f"(checked {smq_candidate}, deoxys {deoxys_config_root})"
    )


def default_osc_controller_cfg_name(controller_type: str) -> str:
    """SMQ-local yaml for teleop / collection."""
    mapping = {
        "OSC_POSE": "configs/deoxys/osc-pose-controller.yml",
        "OSC_POSITION": "configs/deoxys/osc-position-controller.yml",
        "OSC_YAW": "configs/deoxys/osc-yaw-controller.yml",
    }
    return mapping.get(controller_type, "configs/deoxys/osc-pose-controller.yml")
