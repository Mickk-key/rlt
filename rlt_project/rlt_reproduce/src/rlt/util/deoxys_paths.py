"""Resolve deoxys install path under SMQ&JGY (vendored copy preferred)."""

from __future__ import annotations

import os
from pathlib import Path


def smq_root_from_rlt(rlt_root: Path | None = None) -> Path:
    """Return SMQ&JGY repo root (``.../smq&jgy``, parent of ``rlt_project``)."""
    root = Path(rlt_root).resolve() if rlt_root is not None else Path(__file__).resolve().parents[4]
    if root.name == "rlt_reproduce":
        root = root.parent
    if root.name == "rlt_project":
        root = root.parent
    return root


def resolve_deoxys_paths(
    *,
    smq_root: Path | None = None,
    interface_name: str = "charmander.yml",
    robot_cfg: dict | None = None,
) -> tuple[str, str]:
    """Return (deoxys_root, deoxys_config). Prefer ``smq&jgy/third_party/deoxys``."""
    smq = smq_root_from_rlt(smq_root)
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
