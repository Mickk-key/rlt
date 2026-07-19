"""Off-policy replay buffer with chunk + reference-action storage (Algorithm 1)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    """One paper-faithful chunk transition (RLT Eq. 3 / Alg. 1 line 12).

    ``action`` and ``reference_action`` are REAL action chunks of shape
    ``(chunk_length, action_dim)`` — never a single step tiled across the chunk.
    ``next_state`` is the state exactly ``chunk_length`` env-steps after ``state``.
    ``reward`` is the in-chunk discounted return ``sum_k gamma^k r_{t+k}``.
    ``next_reference_action`` is the VLA reference at ``next_state`` (used by the
    critic target ``a' ~ pi(x', ref')``); optional for the legacy single-env path.
    """

    state: np.ndarray
    action: np.ndarray
    reference_action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    next_reference_action: np.ndarray | None = None
    # Fraction/flag of steps in the chunk driven by a human teleop takeover
    # (RLT Sec. V intervention). Metadata only — not used by the TD3 update.
    intervened: float = 0.0


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        """Drop all transitions — use on restart so stale/malformed data never mixes."""
        self.buffer.clear()

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=len(self.buffer) < batch_size)
        return [self.buffer[i] for i in idx]
