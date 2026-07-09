"""Extract VLA reference actions and token embeddings for RLT."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlt.vla.obs_builder import build_observation_for_format
from rlt.vla.openpi_wrapper import MockVLAWrapper, VLAOutput, try_load_openpi_policy

logger = logging.getLogger(__name__)


@dataclass
class VLAInferenceResult:
    reference_action: np.ndarray  # (H, action_dim)
    embeddings: np.ndarray  # (M, embed_dim)


class VLAEmbeddingExtractor:
    """Unified VLA interface for RL token + reference action chunks."""

    def __init__(
        self,
        checkpoint: str | None,
        config_name: str,
        device: str,
        action_dim: int = 7,
        chunk_horizon: int = 50,
        embed_dim: int = 2048,
        num_tokens: int = 64,
        *,
        asset_id: str = "franka",
        default_prompt: str | None = None,
        proprio_mapping: dict[str, Any] | None = None,
        image_size: tuple[int, int] = (224, 224),
        input_format: str = "droid",
        rlt_root: Path | None = None,
    ):
        self.device = device
        self.action_dim = action_dim
        self.chunk_horizon = chunk_horizon
        self.embed_dim = embed_dim
        self.proprio_mapping = proprio_mapping or {}
        self.image_size = image_size
        self.input_format = str(input_format).lower()
        self.default_prompt = default_prompt or ""

        ckpt = checkpoint
        if ckpt and rlt_root is not None:
            ckpt_path = Path(ckpt)
            if not ckpt_path.is_absolute():
                ckpt = str((rlt_root / ckpt_path).resolve())

        self._policy = try_load_openpi_policy(
            ckpt,
            config_name,
            device,
            asset_id=asset_id,
            default_prompt=default_prompt,
            action_dim=action_dim,
            chunk_horizon=chunk_horizon,
        )
        self._is_mock = isinstance(self._policy, MockVLAWrapper)
        if self._is_mock:
            logger.warning(
                "VLA checkpoint unavailable (%s) — using MockVLAWrapper for smoke tests only",
                checkpoint,
            )
            self._policy = MockVLAWrapper(
                action_dim=action_dim,
                chunk_horizon=chunk_horizon,
                embed_dim=embed_dim,
                num_tokens=num_tokens,
            )
        else:
            logger.info("Loaded openpi VLA from %s (config=%s)", ckpt, config_name)

    def infer_from_proprio(
        self,
        proprio: np.ndarray,
        images: dict[str, np.ndarray] | None = None,
        language: str = "",
    ) -> VLAInferenceResult:
        prompt = language or self.default_prompt
        if self._is_mock:
            out: VLAOutput = self._policy.forward(batch_size=1, device=self.device)
            ref = out.reference_action.squeeze(0).cpu().numpy()[: self.chunk_horizon]
            emb = out.embeddings.squeeze(0).cpu().numpy()
            return VLAInferenceResult(reference_action=ref.astype(np.float32), embeddings=emb.astype(np.float32))

        if not images:
            raise ValueError("real openpi VLA requires images_jpeg (external + wrist)")

        openpi_obs = build_observation_for_format(
            self.input_format,
            proprio,
            images,
            prompt,
            proprio_mapping=self.proprio_mapping,
            image_size=self.image_size,
        )
        img_shapes = {
            k: tuple(v.shape)
            for k, v in openpi_obs.items()
            if k.startswith("observation/") and hasattr(v, "shape")
        }
        logger.info("build_openpi_observation keys: %s", img_shapes or "no image tensors")

        # Embed before reference_action: separate JAX JIT compiles peak high on 24GB cards;
        # running embed first avoids holding both compiled graphs from infer+embed at once.
        emb = self._policy.extract_embeddings(openpi_obs)
        ref = self._policy.reference_action(openpi_obs)
        horizon = min(self.chunk_horizon, ref.shape[0])
        padded = np.zeros((self.chunk_horizon, self.action_dim), dtype=np.float32)
        padded[:horizon] = ref[:horizon, : self.action_dim]
        return VLAInferenceResult(reference_action=padded, embeddings=emb.astype(np.float32))
