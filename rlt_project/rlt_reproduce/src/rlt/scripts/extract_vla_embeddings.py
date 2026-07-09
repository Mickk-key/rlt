#!/usr/bin/env python3
"""Extract VLA token embeddings from collected episodes for RL token training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from rich.console import Console
from tqdm import tqdm

from rlt.data.episode_io import list_episodes, load_episode
from rlt.vla.embedding_extractor import VLAEmbeddingExtractor

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/franka/fr3_franka.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    paths = raw["paths"]
    vla_cfg = raw["vla"]
    rt_cfg = raw["rl_token"]
    out_dir = Path(paths["embeddings_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = VLAEmbeddingExtractor(
        checkpoint=vla_cfg.get("checkpoint"),
        config_name=vla_cfg.get("config_name", "pi05_base"),
        device=raw.get("device", "cpu"),
        action_dim=vla_cfg.get("action_dim", 7),
        chunk_horizon=vla_cfg.get("vla_chunk_horizon", 50),
        embed_dim=rt_cfg.get("embed_dim", 2048),
    )

    episodes = list_episodes(paths["episodes_dir"])
    if not episodes:
        console.print("[yellow]No episodes found — run collect_critical_phase.py first[/yellow]")
        return

    console.print(f"Extracting embeddings for {len(episodes)} episodes")
    for ep_path in tqdm(episodes):
        ep = load_episode(ep_path)
        emb_list = []
        for t in range(ep.num_steps()):
            images = {}
            if ep.images_wrist is not None:
                images["wrist"] = ep.images_wrist[t]
            if ep.images_external is not None:
                images["external"] = ep.images_external[t]
            out = extractor.infer_from_proprio(ep.proprio[t], images=images, language=ep.metadata.language)
            emb_list.append(out.embeddings)
        arr = np.stack(emb_list).astype(np.float32)
        out_path = out_dir / f"{ep_path.stem}_embeddings.npz"
        np.savez_compressed(out_path, embeddings=arr, proprio=ep.proprio, language=ep.metadata.language)
        console.print(f"  saved {out_path.name} shape={arr.shape}")


if __name__ == "__main__":
    main()
