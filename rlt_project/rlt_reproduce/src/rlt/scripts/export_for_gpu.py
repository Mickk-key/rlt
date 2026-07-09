#!/usr/bin/env python3
"""Package plug-insertion NPZ episodes for transfer to GPU training host."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/plug_insertion.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    src = Path(raw["paths"]["episodes_dir"])
    dst = Path(raw["paths"]["export_dir"])
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst / "episodes")

    manifest = {
        "task": "plug_insertion",
        "language_instruction": raw["vla"]["language_instruction"],
        "action_dim": raw["vla"]["action_dim"],
        "proprio_dim": raw["robot"]["proprio_dim"],
        "fps": raw["data_collection"]["fps"],
        "phase": raw["data_collection"]["phase"],
        "episodes": sorted(p.name for p in (dst / "episodes").glob("*.npz")),
        "gpu_training_steps": [
            "1. rsync export_dir to GPU host",
            "2. openpi fine-tune (asset_id=franka) on episodes",
            "3. python -m rlt.scripts.extract_vla_embeddings --config configs/plug_insertion.yaml",
            "4. python -m rlt.scripts.train_rl_token --config configs/plug_insertion.yaml",
            "5. copy checkpoints back; run train_online_rl on robot PC",
        ],
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Exported {len(manifest['episodes'])} episodes -> {dst}")
    print(f"Transfer: rsync -avz {dst}/ user@gpu-host:/path/to/rlt_data/plug_insertion/")


if __name__ == "__main__":
    main()
