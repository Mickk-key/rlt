#!/usr/bin/env python3
"""Train RL token encoder-decoder on VLA embeddings (Section IV-A, Eq. 1-2)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from rich.console import Console
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from rlt.rl_token.encoder_decoder import RLTokenEncoderDecoder
from rlt.vla.openpi_wrapper import MockVLAWrapper, try_load_openpi_policy

console = Console()


class EmbeddingShardDataset(Dataset):
    def __init__(self, embeddings_dir: Path):
        self.shards: list[np.ndarray] = []
        for path in sorted(embeddings_dir.glob("*_embeddings.npz")):
            with np.load(path) as data:
                emb = data["embeddings"]  # (T, M, D)
                self.shards.append(emb)
        if not self.shards:
            raise FileNotFoundError(f"No *_embeddings.npz in {embeddings_dir}")

    def __len__(self) -> int:
        return sum(s.shape[0] for s in self.shards)

    def __getitem__(self, idx: int) -> torch.Tensor:
        offset = 0
        for shard in self.shards:
            if idx < offset + shard.shape[0]:
                return torch.as_tensor(shard[idx - offset], dtype=torch.float32)
            offset += shard.shape[0]
        raise IndexError(idx)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/franka/fr3_franka.yaml"))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--embeddings-dir", type=Path, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    rt = raw["rl_token"]
    paths = raw["paths"]
    device = raw.get("device", "cpu")
    steps = args.steps or rt["train_steps"]
    emb_dir = args.embeddings_dir or Path(paths.get("embeddings_dir", "data/embeddings"))

    model = RLTokenEncoderDecoder(
        embed_dim=rt["embed_dim"],
        token_dim=rt["token_dim"],
        num_encoder_layers=rt["num_encoder_layers"],
        num_decoder_layers=rt["num_decoder_layers"],
        num_heads=rt["num_heads"],
        ff_dim=rt["ff_dim"],
        dropout=rt["dropout"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=rt["lr"])

    use_real_emb = emb_dir.exists() and list(emb_dir.glob("*_embeddings.npz"))
    if use_real_emb:
        dataset = EmbeddingShardDataset(emb_dir)
        loader = DataLoader(dataset, batch_size=rt["batch_size"], shuffle=True, drop_last=True)
        console.print(f"[green]Training on embeddings[/green] from {emb_dir} ({len(dataset)} frames)")
        step = 0
        while step < steps:
            for batch in loader:
                batch = batch.to(device)
                loss, _ = model.reconstruction_loss(batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                step += 1
                if step >= steps:
                    break
                if step % 500 == 0:
                    console.print(f"  step {step} L_ro={loss.item():.4f}")
    else:
        vla = try_load_openpi_policy(
            raw["vla"].get("checkpoint"), raw["vla"].get("config_name", "pi05_base"), device
        )
        if not isinstance(vla, MockVLAWrapper):
            console.print("[yellow]Real openpi embedding path not wired — using mock[/yellow]")
            vla = MockVLAWrapper(embed_dim=rt["embed_dim"])
        console.print(f"[yellow]No embeddings in {emb_dir} — mock VLA training[/yellow]")
        for step in tqdm(range(steps)):
            out = vla.forward(batch_size=rt["batch_size"], device=device)
            loss, _ = model.reconstruction_loss(out.embeddings)
            opt.zero_grad()
            loss.backward()
            opt.step()

    ckpt_dir = Path(paths["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "rl_token.pt"
    torch.save(model.state_dict(), ckpt)
    console.print(f"[green]Saved[/green] {ckpt}")


if __name__ == "__main__":
    main()
