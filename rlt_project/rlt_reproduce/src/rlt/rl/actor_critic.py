"""Lightweight actor and critic networks (Section IV-B, Appendix B)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.extend([nn.Linear(prev, h), nn.ReLU()])
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class RLTActor(nn.Module):
    """Bounded-absolute Gaussian actor conditioned on (z_rl, proprio, reference chunk).

    Paper parameterization (Eq. 4): ``pi(a|x, ã) = N(mu(x, ã), sigma^2 I)`` — an
    ABSOLUTE action chunk conditioned on the VLA reference, NOT a residual.

    The mean is bounded to one **global** physical delta-action box via tanh:

        mu = action_box * tanh(net(x, ã))

    This box is a fixed property of the robot's valid delta-action domain
    (e.g. ±0.02 m / ±0.10 rad / gripper range) and is **independent of the VLA
    reference** — it only defines where actions may legally live, so training and
    deployment share the same in-distribution action set (required for TD3
    stability). "Local refinement" toward the VLA is a SEPARATE, SOFT mechanism:
    the BC term ``beta·‖a − ã‖²`` in the actor loss (RLT Eq. 5). The actor is free
    to choose any action inside the box when the Q gain outweighs the BC penalty.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_length: int,
        hidden_dims: list[int],
        fixed_std: float = 0.05,
        action_box: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.chunk_length = chunk_length
        self.action_dim = action_dim
        self.fixed_std = fixed_std
        self._log_std_const = math.log(fixed_std)
        flat_action = chunk_length * action_dim
        self.net = _build_mlp(state_dim + flat_action, hidden_dims, flat_action)

        if action_box is None:
            action_box = [1.0] * action_dim
        if len(action_box) != action_dim:
            raise ValueError(f"action_box len {len(action_box)} != action_dim {action_dim}")
        # (1, 1, action_dim) for broadcasting over (batch, chunk, action_dim).
        # Non-persistent: the box is config-owned, never restored from a checkpoint.
        self.register_buffer(
            "action_box",
            torch.tensor(action_box, dtype=torch.float32).reshape(1, 1, action_dim),
            persistent=False,
        )

    def _net_raw(self, state: torch.Tensor, reference_action: torch.Tensor) -> torch.Tensor:
        """Pre-tanh network output (unbounded) — exposed for diagnostics only."""
        x = torch.cat([state, reference_action.reshape(state.shape[0], -1)], dim=-1)
        return self.net(x).reshape(-1, self.chunk_length, self.action_dim)

    def _bound(self, action: torch.Tensor) -> torch.Tensor:
        """Clamp per-dim to ±action_box (broadcast), keeping samples inside the box."""
        box = self.action_box
        return torch.minimum(torch.maximum(action, -box), box)

    def forward(
        self,
        state: torch.Tensor,
        reference_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self._net_raw(state, reference_action)
        mean = self.action_box * torch.tanh(raw)  # bounded absolute output
        log_std = torch.full_like(mean, fill_value=self._log_std_const)
        return mean, log_std

    def sample(
        self,
        state: torch.Tensor,
        reference_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state, reference_action)
        std = log_std.exp()
        action = mean + torch.randn_like(mean) * std
        action = self._bound(action)  # sampled actions must also stay inside the box
        log_prob = -0.5 * (((action - mean) / std).pow(2) + 2 * log_std + math.log(2 * math.pi))
        return action, log_prob.sum(dim=(-1, -2))


class RLTCritic(nn.Module):
    """Q(s, a) over flattened action chunks."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_length: int,
        hidden_dims: list[int],
    ) -> None:
        super().__init__()
        flat_action = chunk_length * action_dim
        self.net = _build_mlp(state_dim + flat_action, hidden_dims, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action.reshape(state.shape[0], -1)], dim=-1)
        return self.net(x).squeeze(-1)


class RLTCriticEnsemble(nn.Module):
    """Twin Q-networks (TD3-style, Appendix B)."""

    def __init__(self, num_q: int, **kwargs) -> None:
        super().__init__()
        self.qs = nn.ModuleList([RLTCritic(**kwargs) for _ in range(num_q)])

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.stack([q(state, action) for q in self.qs], dim=0)

    def min_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.forward(state, action).min(dim=0).values
