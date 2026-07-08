#!/usr/bin/env python3
"""Ping GPU RL websocket server from the robot PC."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml
from rich.console import Console

from rlt.rl.gpu_client import create_gpu_client

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description="Health-check GPU RL websocket server")
    parser.add_argument("--config", type=Path, default=Path("configs/plug_insertion.yaml"))
    parser.add_argument("--host", default=None, help="Override gpu_server.host / GPU_SERVER_HOST")
    parser.add_argument("--mock", action="store_true", help="Ping local MockGPUClient")
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    host = args.host or os.environ.get("GPU_SERVER_HOST")
    client = create_gpu_client(raw, mock=args.mock, host_override=host)
    try:
        resp = client.ping()
    except Exception as exc:
        console.print(f"[red]FAIL[/red] {exc}")
        return 1
    finally:
        client.close()

    console.print(f"[green]OK[/green] {resp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
