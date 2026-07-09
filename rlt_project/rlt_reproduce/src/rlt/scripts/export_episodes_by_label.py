#!/usr/bin/env python3
"""Export plug-insertion episodes into success/ and fail/ folders + manifest."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from rlt.data.episode_io import list_episodes


def _episode_stats(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"]))
        timestamps = data["timestamps"]
        dur = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
        proprio = data["proprio"]
        xyz_ok = bool(np.linalg.norm(proprio[:, :3], axis=1).max() > 0.1)
        return {
            "file": path.name,
            "success": bool(meta.get("success", False)),
            "steps": int(len(timestamps)),
            "duration_sec": round(dur, 2),
            "xyz_ok": xyz_ok,
            "task": meta.get("task"),
            "phase": meta.get("phase"),
        }


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split episodes by success/fail label.")
    parser.add_argument("--config", type=Path, default=Path("configs/plug_insertion.yaml"))
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Output root (default: data/export/plug_insertion under RLT root)",
    )
    parser.add_argument(
        "--skip-long-sec",
        type=float,
        default=0.0,
        help="Skip episodes longer than this (0 = keep all)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    rlt_root = args.config.resolve().parents[1]
    episodes_dir = Path(raw["paths"]["episodes_dir"])
    if not episodes_dir.is_absolute():
        episodes_dir = rlt_root / episodes_dir

    export_root = args.export_dir or (rlt_root / "data" / "export" / "plug_insertion")
    export_root = export_root.resolve()

    for sub in ("success", "fail"):
        subdir = export_root / sub
        if subdir.exists():
            shutil.rmtree(subdir)

    episodes = list_episodes(episodes_dir)
    if not episodes:
        raise SystemExit(f"No episodes in {episodes_dir}")

    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(episodes_dir),
        "export_dir": str(export_root),
        "skip_long_sec": args.skip_long_sec,
        "counts": {"success": 0, "fail": 0, "skipped": 0},
        "episodes": {"success": [], "fail": [], "skipped": []},
    }

    for path in episodes:
        stats = _episode_stats(path)
        if args.skip_long_sec > 0 and stats["duration_sec"] > args.skip_long_sec:
            manifest["counts"]["skipped"] += 1
            manifest["episodes"]["skipped"].append({**stats, "reason": "too_long"})
            continue

        label = "success" if stats["success"] else "fail"
        dest = export_root / label / path.name
        _link_or_copy(path, dest)
        manifest["counts"][label] += 1
        manifest["episodes"][label].append(stats)

    manifest_path = export_root / "manifest.json"
    export_root.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    readme = export_root / "README.txt"
    readme.write_text(
        "Plug insertion episodes exported by success/fail label.\n"
        f"  success/: {manifest['counts']['success']} episodes\n"
        f"  fail/:    {manifest['counts']['fail']} episodes\n"
        f"  skipped: {manifest['counts']['skipped']} (see manifest.json)\n"
        f"  source:  {episodes_dir}\n",
        encoding="utf-8",
    )

    total = manifest["counts"]["success"] + manifest["counts"]["fail"]
    print(f"Exported {total} episodes -> {export_root}")
    print(f"  success: {manifest['counts']['success']}")
    print(f"  fail:    {manifest['counts']['fail']}")
    if manifest["counts"]["skipped"]:
        print(f"  skipped: {manifest['counts']['skipped']} (>{args.skip_long_sec}s)")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
