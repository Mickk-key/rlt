#!/usr/bin/env python3
"""Batch-convert SFT JSONL episode folders to NPZ for legacy RLT tools."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from rich.console import Console

from rlt.data.sft_io import export_sft_episode_to_npz, list_sft_episodes

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sft_plug_insertion.yaml"))
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    src = args.input_dir or Path(raw.get("sft_collection", {}).get("output_dir", "data/sft/plug_insertion"))
    dst = args.output_dir or Path(raw["paths"].get("episodes_dir", "data/episodes/plug_insertion"))
    dst.mkdir(parents=True, exist_ok=True)

    episodes = list_sft_episodes(src)
    if not episodes:
        console.print(f"[yellow]No SFT episodes under {src}[/yellow]")
        return

    for ep_dir in episodes:
        out = dst / f"{ep_dir.name}.npz"
        export_sft_episode_to_npz(ep_dir, out_path=out)
        console.print(f"[green]Exported[/green] {ep_dir.name} → {out.name}")

    console.print(f"Done: {len(episodes)} episodes → {dst}")


if __name__ == "__main__":
    main()
