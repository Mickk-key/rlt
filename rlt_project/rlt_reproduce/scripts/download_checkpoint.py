#!/usr/bin/env python3
"""Download openpi VLA checkpoint (requires gsutil or openpi download helper)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CHECKPOINTS = {
    "pi05_base": "gs://openpi-assets/checkpoints/pi05_base",
    "pi05_libero": "gs://openpi-assets/checkpoints/pi05_libero",
    "pi0_droid": "gs://openpi-assets/checkpoints/pi0_droid",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi05_base", choices=list(CHECKPOINTS))
    parser.add_argument("--out", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    out = args.out / args.config
    out.mkdir(parents=True, exist_ok=True)
    uri = CHECKPOINTS[args.config]

    print(f"Downloading {uri} -> {out}")
    try:
        from openpi.shared import download

        path = download.maybe_download(uri, out)
        print(f"Done: {path}")
    except ImportError:
        print("openpi not installed. Fallback: gsutil -m cp -r", uri, str(out))
        subprocess.check_call(["gsutil", "-m", "cp", "-r", uri, str(out)])


if __name__ == "__main__":
    main()
