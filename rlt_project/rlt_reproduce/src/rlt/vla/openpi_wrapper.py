"""openpi VLA wrapper: reference actions + prefix embeddings for RL token."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch

from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.policies import policy_config
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as train_config


@dataclass
class VLAOutput:
    reference_action: torch.Tensor
    embeddings: torch.Tensor


class MockVLAWrapper:
    def __init__(self, action_dim: int = 7, chunk_horizon: int = 15, embed_dim: int = 2048, num_tokens: int = 64):
        self.action_dim = action_dim
        self.chunk_horizon = chunk_horizon
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

    def forward(self, batch_size: int = 1, device: str = "cpu") -> VLAOutput:
        ref = torch.randn(batch_size, self.chunk_horizon, self.action_dim, device=device) * 0.1
        emb = torch.randn(batch_size, self.num_tokens, self.embed_dim, device=device)
        return VLAOutput(reference_action=ref, embeddings=emb)


class OpenPIVLAWrapper:
    """Real openpi policy with embedding extraction and action inference."""

    def __init__(self, policy: Any, *, action_dim: int = 7, chunk_horizon: int = 15) -> None:
        self.policy = policy
        self.action_dim = action_dim
        self.chunk_horizon = chunk_horizon
        self._embed_fn = None

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        config_name: str,
        *,
        asset_id: str = "franka",
        default_prompt: str | None = None,
        device: str = "cuda",
        action_dim: int = 7,
        chunk_horizon: int = 15,
    ) -> OpenPIVLAWrapper:
        cfg = train_config.get_config(config_name)
        ckpt = Path(checkpoint)
        norm_stats = _checkpoints.load_norm_stats(ckpt / "assets", asset_id)
        policy = policy_config.create_trained_policy(
            cfg,
            ckpt,
            default_prompt=default_prompt,
            norm_stats=norm_stats,
            pytorch_device=device,
        )
        return cls(policy, action_dim=action_dim, chunk_horizon=chunk_horizon)

    def _build_embed_fn(self):
        if self._embed_fn is not None:
            return self._embed_fn
        model = self.policy._model

        def _extract(inputs: dict) -> jnp.ndarray:
            observation = _model.Observation.from_dict(inputs)
            observation = _model.preprocess_observation(None, observation, train=False)
            prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            (prefix_out, _), _ = model.PaliGemma.llm(
                [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
            )
            return prefix_out

        self._embed_fn = jax.jit(_extract)
        return self._embed_fn

    def infer_numpy(self, obs: dict[str, Any]) -> dict[str, Any]:
        return self.policy.infer(obs)

    def extract_embeddings(self, obs: dict[str, Any]) -> np.ndarray:
        """Return (M, embed_dim) prefix token embeddings."""
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self.policy._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        prefix_out = self._build_embed_fn()(inputs)
        return np.asarray(prefix_out[0], dtype=np.float32)

    def reference_action(self, obs: dict[str, Any]) -> np.ndarray:
        out = self.infer_numpy(obs)
        actions = np.asarray(out["actions"], dtype=np.float32)
        return actions[: self.chunk_horizon, : self.action_dim]

    def forward_batch(self, obs_list: list[dict[str, Any]], device: str = "cuda") -> VLAOutput:
        refs = []
        embs = []
        for obs in obs_list:
            ref = self.reference_action(obs)
            emb = self.extract_embeddings(obs)
            embs.append(emb)
            horizon = min(self.chunk_horizon, ref.shape[0])
            padded = np.zeros((self.chunk_horizon, self.action_dim), dtype=np.float32)
            padded[:horizon] = ref[:horizon]
            refs.append(padded)
        ref_t = torch.as_tensor(np.stack(refs), device=device)
        emb_t = torch.as_tensor(np.stack(embs), device=device)
        return VLAOutput(reference_action=ref_t, embeddings=emb_t)


def try_load_openpi_policy(
    checkpoint: str | None,
    config_name: str,
    device: str,
    *,
    asset_id: str = "franka",
    default_prompt: str | None = None,
    action_dim: int = 7,
    chunk_horizon: int = 15,
) -> Any:
    if checkpoint is None:
        return MockVLAWrapper(action_dim=action_dim, chunk_horizon=chunk_horizon)
    try:
        return OpenPIVLAWrapper.load(
            checkpoint,
            config_name,
            asset_id=asset_id,
            default_prompt=default_prompt,
            device=device,
            action_dim=action_dim,
            chunk_horizon=chunk_horizon,
        )
    except ImportError as exc:
        raise ImportError(
            "openpi is not installed. Run: source scripts/activate_rlt.sh"
        ) from exc
