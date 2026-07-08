"""Optional wrapper around openpi VLA for RL token extraction and reference actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class VLAOutput:
    reference_action: torch.Tensor  # (B, H, action_dim)
    embeddings: torch.Tensor  # (B, M, embed_dim)


class MockVLAWrapper:
    """CPU-friendly stand-in until openpi checkpoint + GPU are available."""

    def __init__(self, action_dim: int = 14, chunk_horizon: int = 50, embed_dim: int = 2048, num_tokens: int = 64):
        self.action_dim = action_dim
        self.chunk_horizon = chunk_horizon
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

    def forward(self, batch_size: int = 1, device: str = "cpu") -> VLAOutput:
        ref = torch.randn(batch_size, self.chunk_horizon, self.action_dim, device=device) * 0.1
        emb = torch.randn(batch_size, self.num_tokens, self.embed_dim, device=device)
        return VLAOutput(reference_action=ref, embeddings=emb)


def try_load_openpi_policy(checkpoint: str | None, config_name: str, device: str) -> Any:
    """Load openpi policy when available; otherwise return MockVLAWrapper."""
    if checkpoint is None:
        return MockVLAWrapper()

    try:
        from openpi.policies import policy_config
        from openpi.training import config as train_config

        cfg = train_config.get_config(config_name)
        policy = policy_config.create_trained_policy(cfg, checkpoint, pytorch_device=device)
        return policy
    except ImportError as exc:
        raise ImportError(
            "openpi is not installed. Run scripts/setup_env.sh first, "
            "or use MockVLAWrapper for CPU smoke tests."
        ) from exc
