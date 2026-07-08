"""Configuration loading for RLT reproduction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RLTConfig:
    vla: dict[str, Any] = field(default_factory=dict)
    rl_token: dict[str, Any] = field(default_factory=dict)
    online_rl: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"

    @classmethod
    def from_yaml(cls, path: str | Path) -> RLTConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def ensure_dirs(self) -> None:
        for key in ("data_dir", "checkpoint_dir", "log_dir", "output_dir"):
            Path(self.paths[key]).mkdir(parents=True, exist_ok=True)
