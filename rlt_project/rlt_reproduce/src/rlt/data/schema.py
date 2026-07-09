"""RLT dataset schema for real-robot collection and training."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpisodeMetadata:
    task: str
    language: str
    phase: str  # "critical" | "full"
    success: bool
    robot: str = "fr3_franka"
    fps: float = 20.0
    action_dim: int = 7
    proprio_dim: int = 8
    extra: dict = field(default_factory=dict)


@dataclass
class CriticalPhaseEpisode:
    """One recorded episode for RLT (paper Section IV).

    Stored on disk as NPZ + optional JPEG folder per episode.
    """

    metadata: EpisodeMetadata
    timestamps: np.ndarray  # (T,)
    proprio: np.ndarray  # (T, proprio_dim) ee + gripper
    actions: np.ndarray  # (T, action_dim) executed commands
    rewards: np.ndarray  # (T,) sparse step rewards if labeled
    dones: np.ndarray  # (T,) bool
    is_human: np.ndarray  # (T,) bool human intervention flags
    reference_actions: np.ndarray | None = None  # (T, H, action_dim) if VLA logged online
    images_wrist: np.ndarray | None = None  # (T, H, W, 3) uint8
    images_external: np.ndarray | None = None  # (T, H, W, 3) uint8

    def num_steps(self) -> int:
        return int(self.timestamps.shape[0])


@dataclass
class EmbeddingRecord:
    """VLA final-layer token embeddings for RL token training (Eq. 1-2)."""

    episode_id: str
    embeddings: np.ndarray  # (T, M, embed_dim)
    proprio: np.ndarray  # (T, proprio_dim) optional auxiliary
    language: str = ""


@dataclass
class RLTransitionRecord:
    """Chunk-level transition for online RL replay (Algorithm 1)."""

    state: np.ndarray  # concat(z_rl, proprio)
    action: np.ndarray  # (C, action_dim)
    reference_action: np.ndarray  # (C, action_dim)
    reward: float
    next_state: np.ndarray
    done: bool
    is_human: bool = False
