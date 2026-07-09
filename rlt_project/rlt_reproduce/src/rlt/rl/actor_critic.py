"""Lightweight actor and critic networks (Section IV-B, Appendix B)."""

from __future__ import annotations

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
    """Gaussian actor conditioned on (z_rl, proprio, reference action chunk)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_length: int,
        hidden_dims: list[int],
        fixed_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.chunk_length = chunk_length
        self.action_dim = action_dim
        self.fixed_std = fixed_std
        flat_action = chunk_length * action_dim
        self.net = _build_mlp(state_dim + flat_action, hidden_dims, flat_action)

    def forward(
        self,
        state: torch.Tensor,
        reference_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, reference_action.reshape(state.shape[0], -1)], dim=-1)
        mean = self.net(x).reshape(-1, self.chunk_length, self.action_dim)
        log_std = torch.full_like(mean, fill_value=torch.log(torch.tensor(self.fixed_std)))
        return mean, log_std

    def sample(
        self,
        state: torch.Tensor,
        reference_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state, reference_action)
        std = log_std.exp()
        noise = torch.randn_like(mean)
        action = mean + noise * std
        log_prob = -0.5 * (((action - mean) / std).pow(2) + 2 * log_std + torch.log(torch.tensor(2 * 3.14159265)))
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
