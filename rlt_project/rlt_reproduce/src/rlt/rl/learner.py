"""TD3-style off-policy learner for RLT (Eq. 3-5, Algorithm 1)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from rlt.rl.actor_critic import RLTActor, RLTCriticEnsemble
from rlt.rl.replay_buffer import ReplayBuffer, Transition


@dataclass
class LearnerMetrics:
    critic_loss: float
    actor_loss: float


class RLTLearner:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_length: int,
        actor_hidden: list[int],
        critic_hidden: list[int],
        *,
        device: str = "cpu",
        discount: float = 0.99,
        policy_constraint_beta: float = 1.0,
        reference_dropout: float = 0.5,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        critic_ensemble: int = 2,
        target_tau: float = 0.005,
    ) -> None:
        self.device = torch.device(device)
        self.discount = discount
        self.chunk_length = chunk_length
        self.policy_constraint_beta = policy_constraint_beta
        self.reference_dropout = reference_dropout

        ac_kwargs = dict(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_length=chunk_length,
        )
        self.actor = RLTActor(**ac_kwargs, hidden_dims=actor_hidden).to(self.device)
        self.actor_target = deepcopy(self.actor)
        self.critic = RLTCriticEnsemble(num_q=critic_ensemble, **ac_kwargs, hidden_dims=critic_hidden).to(self.device)
        self.critic_target = deepcopy(self.critic)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.target_tau = target_tau

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    def _batchify(self, batch: list[Transition]) -> dict[str, torch.Tensor]:
        # ``next_reference`` = VLA reference at the *next* state (ref'), needed for the
        # paper target a' ~ pi(x', ref'). Legacy single-env transitions omit it, so we
        # fall back to the current reference for backward compatibility.
        next_ref = [
            t.next_reference_action if t.next_reference_action is not None else t.reference_action
            for t in batch
        ]
        return {
            "state": self._to_tensor(np.stack([t.state for t in batch])),
            "action": self._to_tensor(np.stack([t.action for t in batch])),
            "reference": self._to_tensor(np.stack([t.reference_action for t in batch])),
            "next_reference": self._to_tensor(np.stack(next_ref)),
            "reward": self._to_tensor(np.array([t.reward for t in batch])),
            "next_state": self._to_tensor(np.stack([t.next_state for t in batch])),
            "done": self._to_tensor(np.array([float(t.done) for t in batch])),
        }

    def update_critic(self, batch: list[Transition]) -> float:
        data = self._batchify(batch)
        with torch.no_grad():
            # a' ~ pi(x', ref') — condition the next action on the NEXT state's reference.
            next_action, _ = self.actor_target.sample(data["next_state"], data["next_reference"])
            target_q = self.critic_target.min_q(data["next_state"], next_action)
            # Temporal gap of every stored transition is exactly ``chunk_length`` env steps
            # (next_state = x_{t+C}), so the bootstrap discount is gamma^C — matching the
            # in-chunk return R = sum_{k=0}^{C-1} gamma^k r_{t+k} already stored in reward.
            gamma_c = self.discount ** self.chunk_length
            q_target = data["reward"] + (1.0 - data["done"]) * gamma_c * target_q

        qs = self.critic(data["state"], data["action"])
        loss = F.mse_loss(qs, q_target.unsqueeze(0).expand_as(qs))
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()
        return float(loss.item())

    def update_actor(self, batch: list[Transition]) -> float:
        data = self._batchify(batch)
        ref = data["reference"].clone()
        if self.reference_dropout > 0:
            mask = torch.rand(ref.shape[0], device=ref.device) < self.reference_dropout
            ref[mask] = 0.0

        action, _ = self.actor.sample(data["state"], ref)
        q = self.critic.min_q(data["state"], action)
        bc_penalty = (action - data["reference"]).pow(2).mean(dim=(-1, -2))
        loss = (-q + self.policy_constraint_beta * bc_penalty).mean()

        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        return float(loss.item())

    def soft_update_targets(self) -> None:
        for src, tgt in (
            (self.actor, self.actor_target),
            (self.critic, self.critic_target),
        ):
            for p, tp in zip(src.parameters(), tgt.parameters()):
                tp.data.mul_(1 - self.target_tau).add_(p.data, alpha=self.target_tau)

    def train_step(
        self,
        buffer: ReplayBuffer,
        batch_size: int,
        critic_updates: int = 2,
    ) -> LearnerMetrics:
        batch = buffer.sample(batch_size)
        critic_loss = 0.0
        for _ in range(critic_updates):
            critic_loss = self.update_critic(batch)
        actor_loss = self.update_actor(batch)
        self.soft_update_targets()
        return LearnerMetrics(critic_loss=critic_loss, actor_loss=actor_loss)
