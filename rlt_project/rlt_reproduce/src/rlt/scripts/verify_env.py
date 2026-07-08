#!/usr/bin/env python3
"""Verify RLT reproduction environment (CPU-safe, no GPU required)."""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
    console.print(f"  {status} {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    console.print(f"\n[bold]RLT environment check[/bold] ({root})\n")

    results: list[bool] = []
    results.append(check("Python >= 3.11", sys.version_info >= (3, 11), sys.version))
    results.append(check("Platform", True, platform.platform()))

    for pkg in ("torch", "numpy", "yaml", "rich", "einops"):
        try:
            mod = importlib.import_module(pkg if pkg != "yaml" else "yaml")
            ver = getattr(mod, "__version__", "?")
            results.append(check(f"import {pkg}", True, ver))
        except ImportError:
            results.append(check(f"import {pkg}", False))

    import torch

    cuda_ok = torch.cuda.is_available()
    check("CUDA available", cuda_ok, "ready" if cuda_ok else "optional until RTX 5090 + driver")

    # Core RLT modules
    try:
        from rlt.rl_token.encoder_decoder import RLTokenEncoderDecoder
        from rlt.rl.learner import RLTLearner

        model = RLTokenEncoderDecoder(embed_dim=64, token_dim=64, num_encoder_layers=1, num_decoder_layers=1, num_heads=4, ff_dim=128)
        x = torch.randn(2, 8, 64)
        loss, z = model.reconstruction_loss(x)
        results.append(check("RL token forward", loss.ndim == 0 and z.shape == (2, 64), f"loss={loss.item():.4f}"))

        learner = RLTLearner(state_dim=64, action_dim=4, chunk_length=2, actor_hidden=[32], critic_hidden=[32], device="cpu")
        results.append(check("RL learner init", True))
    except Exception as exc:
        results.append(check("RLT core modules", False, str(exc)))

    # openpi (optional until setup_env.sh completes)
    try:
        import openpi  # noqa: F401

        results.append(check("openpi installed", True))
    except ImportError:
        results.append(check("openpi installed", False, "run scripts/setup_env.sh"))

    table = Table(title="Summary")
    table.add_column("Item")
    table.add_column("Status")
    table.add_row("Required checks", "[green]PASS[/green]" if all(results[:6]) else "[red]FAIL[/red]")
    table.add_row("GPU inference", "[yellow]SKIP (no GPU)[/yellow]" if not cuda_ok else "[green]READY[/green]")
    table.add_row("openpi VLA", "[yellow]PENDING[/yellow]" if "openpi" not in sys.modules else "[green]READY[/green]")
    console.print(table)

    if not all(results[:6]):
        console.print("\n[red]Fix failed checks, then re-run: rlt-verify[/red]")
        sys.exit(1)
    console.print("\n[green]Environment ready for CPU development.[/green] After GPU install, run scripts/setup_gpu.sh\n")


if __name__ == "__main__":
    main()
