#!/usr/bin/env python3
"""Export frame-0 initial states from NPZ episodes to JSON for demo-driven reset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def export_npz_dir(src_dir: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for npz in sorted(src_dir.glob("*.npz")):
        with np.load(npz, allow_pickle=False) as data:
            proprio0 = np.asarray(data["proprio"][0], dtype=float)
            meta = {}
            if "metadata_json" in data:
                meta = json.loads(str(data["metadata_json"]))
        if proprio0.shape[0] < 8 or np.linalg.norm(proprio0[:3]) < 1e-4:
            continue
        payload = {
            "episode_id": npz.stem,
            "success": bool(meta.get("success", False)),
            "initial_state": {
                "ee_pose": proprio0[:3].tolist(),
                "quaternion": proprio0[3:7].tolist(),
                "gripper_width": float(proprio0[7]),
            },
        }
        out_path = out_dir / f"{npz.stem}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/episodes/plug_insertion"),
        help="Directory containing NPZ episodes",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/plug_insertion"),
        help="Output directory for per-episode JSON initial states",
    )
    args = parser.parse_args()
    n = export_npz_dir(args.src.resolve(), args.out.resolve())
    print(f"Exported {n} initial-state JSON files -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
