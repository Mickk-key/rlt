"""Extract VLA reference actions and token embeddings for RLT."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rlt.vla.openpi_wrapper import MockVLAWrapper, VLAOutput, try_load_openpi_policy

# Robot actor sends JPEG-decoded dict keys ``external`` + ``wrist`` (224×224 RGB uint8).
# Default openpi pi05 DROID-style policy expects these observation keys:
#   observation/exterior_image_1_left  ← external
#   observation/wrist_image_left       ← wrist
#   observation/joint_position (7,) + observation/gripper_position (1,)
IMAGE_KEY_MAP = {
    "external": "observation/exterior_image_1_left",
    "wrist": "observation/wrist_image_left",
}


def build_openpi_observation(
    proprio: np.ndarray,
    images: dict[str, np.ndarray] | None,
    language: str,
) -> dict:
    """Map robot-side obs to openpi policy.infer() dict."""
    if images is None or not images:
        raise ValueError(
            "VLA infer requires images_jpeg (decoded wrist + external). "
            "Robot actor sent proprio only — check RealSense / actor_loop."
        )
    missing = [k for k in IMAGE_KEY_MAP if k not in images]
    if missing:
        raise ValueError(
            f"VLA infer missing camera keys {missing}; got {sorted(images)}. "
            f"Expected {sorted(IMAGE_KEY_MAP)}."
        )
    proprio = np.asarray(proprio, dtype=np.float32).reshape(-1)
    if proprio.shape[0] < 8:
        raise ValueError(f"proprio must be 8-d (7 joints + gripper); got shape {proprio.shape}")
    obs: dict = {
        IMAGE_KEY_MAP["external"]: np.asarray(images["external"], dtype=np.uint8),
        IMAGE_KEY_MAP["wrist"]: np.asarray(images["wrist"], dtype=np.uint8),
        "observation/joint_position": proprio[:7],
        "observation/gripper_position": proprio[7:8],
    }
    if language:
        obs["prompt"] = language
    return obs


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
    ):
        self.device = device
        self.action_dim = action_dim
        self.chunk_horizon = chunk_horizon
        self._policy = try_load_openpi_policy(checkpoint, config_name, device)
        self._is_mock = isinstance(self._policy, MockVLAWrapper)
        if self._is_mock:
            self._policy = MockVLAWrapper(
                action_dim=action_dim,
                chunk_horizon=chunk_horizon,
                embed_dim=embed_dim,
                num_tokens=num_tokens,
            )

    def infer_from_proprio(
        self,
        proprio: np.ndarray,
        images: dict[str, np.ndarray] | None = None,
        language: str = "",
    ) -> VLAInferenceResult:
        """Run VLA (or mock) to get reference chunk + embeddings."""
        if self._is_mock:
            del images, language
            out: VLAOutput = self._policy.forward(batch_size=1, device=self.device)
            ref = out.reference_action.squeeze(0).cpu().numpy()[: self.chunk_horizon]
            emb = out.embeddings.squeeze(0).cpu().numpy()
            return VLAInferenceResult(reference_action=ref.astype(np.float32), embeddings=emb.astype(np.float32))

        obs = build_openpi_observation(proprio, images, language)
        result = self._policy.infer(obs)
        actions = np.asarray(result["actions"], dtype=np.float32)
        ref = actions[: self.chunk_horizon]
        # TODO: hook intermediate embeddings from openpi for RL token encoder.
        # Until wired, use zero embeddings of expected shape so RL token path runs.
        embed_dim = getattr(self._policy, "_embed_dim", 2048)
        num_tokens = 64
        emb = np.zeros((num_tokens, embed_dim), dtype=np.float32)
        return VLAInferenceResult(reference_action=ref, embeddings=emb)

    def reference_chunk(self, proprio: np.ndarray, execute_prefix: int) -> np.ndarray:
        result = self.infer_from_proprio(proprio)
        c = min(execute_prefix, result.reference_action.shape[0])
        return result.reference_action[:c]

    def encode_rl_state(
        self,
        token_model: torch.nn.Module,
        proprio: np.ndarray,
        images: dict[str, np.ndarray] | None = None,
        language: str = "",
    ) -> np.ndarray:
        vla = self.infer_from_proprio(proprio, images=images, language=language)
        with torch.no_grad():
            emb = torch.as_tensor(vla.embeddings, device=self.device).unsqueeze(0)
            z = token_model.encode(emb).squeeze(0).cpu().numpy()
        return np.concatenate([z, proprio.astype(np.float32)])
