"""Minimal mock environment for offline algorithm testing without a robot."""

from __future__ import annotations

import numpy as np


class MockPrecisionEnv:
    """Toy env: agent must refine VLA reference toward a hidden target."""

    def __init__(self, state_dim: int, action_dim: int, chunk_length: int, seed: int = 0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_length = chunk_length
        self.rng = np.random.default_rng(seed)
        self.target = self.rng.normal(size=(chunk_length, action_dim)).astype(np.float32)
        self.state = np.zeros(state_dim, dtype=np.float32)
        self.step_count = 0

    def reset(self) -> np.ndarray:
        self.state = self.rng.normal(size=(self.state_dim,)).astype(np.float32) * 0.1
        self.step_count = 0
        return self.state.copy()

    def reference_action(self) -> np.ndarray:
        noise = self.rng.normal(scale=0.5, size=self.target.shape).astype(np.float32)
        return (self.target + noise).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        err = float(np.linalg.norm(action - self.target))
        reward = 1.0 if err < 0.5 else 0.0
        self.state = self.rng.normal(size=(self.state_dim,)).astype(np.float32) * 0.1
        self.step_count += 1
        done = reward > 0 or self.step_count >= 20
        return self.state.copy(), reward, done, {"error": err}
