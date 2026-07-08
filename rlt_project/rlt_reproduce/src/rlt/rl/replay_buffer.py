"""Off-policy replay buffer with reference-action storage (Algorithm 1)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    reference_action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=len(self.buffer) < batch_size)
        return [self.buffer[i] for i in idx]
