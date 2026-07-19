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
    bc_loss: float = 0.0
    q_mean: float = 0.0
    q_abs_max: float = 0.0
    target_q_mean: float = 0.0
    raw_actor_norm: float = 0.0          # pre-tanh network magnitude (saturation diag)
    bounded_actor_norm: float = 0.0      # actual (in-box) actor action magnitude
    target_action_norm: float = 0.0      # smoothed+clamped TD-target action magnitude
    nan_inf: float = 0.0                 # 1.0 if any loss was non-finite this step


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
        action_box: list[float] | None = None,
        target_policy_noise: float = 0.2,   # TD3 target smoothing std, as fraction of action_box
        target_noise_clip: float = 0.5,     # TD3 target smoothing clip, as fraction of action_box
        grad_clip_norm: float = 10.0,       # max global grad norm (defense-in-depth)
    ) -> None:
        self.device = torch.device(device)
        self.discount = discount
        self.chunk_length = chunk_length
        self.policy_constraint_beta = policy_constraint_beta
        self.reference_dropout = reference_dropout
        self.target_policy_noise = float(target_policy_noise)
        self.target_noise_clip = float(target_noise_clip)
        self.grad_clip_norm = float(grad_clip_norm)

        ac_kwargs = dict(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_length=chunk_length,
        )
        self.actor = RLTActor(**ac_kwargs, hidden_dims=actor_hidden, action_box=action_box).to(self.device)
        self.actor_target = deepcopy(self.actor)
        self.critic = RLTCriticEnsemble(num_q=critic_ensemble, **ac_kwargs, hidden_dims=critic_hidden).to(self.device)
        self.critic_target = deepcopy(self.critic)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.target_tau = target_tau

        # (1, 1, action_dim) global physical delta-action box for TD3 target clamping.
        box = list(action_box) if action_box is not None else [1.0] * action_dim
        self.action_box = torch.tensor(
            [float(b) for b in box], dtype=torch.float32, device=self.device
        ).reshape(1, 1, action_dim)

    def _bound_action(self, action: torch.Tensor) -> torch.Tensor:
        return torch.minimum(torch.maximum(action, -self.action_box), self.action_box)

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

    def update_critic(self, batch: list[Transition]) -> tuple[float, dict[str, float]]:
        data = self._batchify(batch)
        with torch.no_grad():
            # a' ~ pi(x', ref') — condition the next action on the NEXT state's reference,
            # then apply TD3 target-policy smoothing: clipped Gaussian noise on the
            # deterministic target mean, and clamp the result to the SAME global box so
            # the target critic is NEVER evaluated on out-of-box actions.
            next_mean, _ = self.actor_target.forward(data["next_state"], data["next_reference"])
            box = self.action_box
            raw_noise = torch.randn_like(next_mean) * (self.target_policy_noise * box)
            clip = self.target_noise_clip * box
            noise = torch.minimum(torch.maximum(raw_noise, -clip), clip)
            next_action = self._bound_action(next_mean + noise)
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
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_opt.step()
        diag = {
            "q_mean": float(qs.mean().item()),
            "q_abs_max": float(qs.abs().max().item()),
            "target_q_mean": float(q_target.mean().item()),
            "target_action_norm": float(next_action.norm(dim=(-1, -2)).mean().item()),
        }
        return float(loss.item()), diag

    def update_actor(self, batch: list[Transition]) -> tuple[float, dict[str, float]]:
        data = self._batchify(batch)
        ref = data["reference"].clone()
        if self.reference_dropout > 0:
            mask = torch.rand(ref.shape[0], device=ref.device) < self.reference_dropout
            ref[mask] = 0.0

        action, _ = self.actor.sample(data["state"], ref)  # already bounded to the box
        q = self.critic.min_q(data["state"], action)
        # SOFT local refinement (RLT Eq. 5): regularize toward the TRUE VLA reference
        # (never the dropped-out copy), independent of the hard global box above.
        bc_penalty = (action - data["reference"]).pow(2).mean(dim=(-1, -2))
        loss = (-q + self.policy_constraint_beta * bc_penalty).mean()

        self.actor_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        self.actor_opt.step()
        with torch.no_grad():
            raw = self.actor._net_raw(data["state"], ref)
        diag = {
            "bc_loss": float(bc_penalty.mean().item()),
            "raw_actor_norm": float(raw.norm(dim=(-1, -2)).mean().item()),
            "bounded_actor_norm": float(action.norm(dim=(-1, -2)).mean().item()),
        }
        return float(loss.item()), diag

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
        cdiag: dict[str, float] = {}
        for _ in range(critic_updates):
            critic_loss, cdiag = self.update_critic(batch)
        actor_loss, adiag = self.update_actor(batch)
        self.soft_update_targets()
        nan_inf = not (np.isfinite(critic_loss) and np.isfinite(actor_loss))
        return LearnerMetrics(
            critic_loss=critic_loss,
            actor_loss=actor_loss,
            bc_loss=adiag.get("bc_loss", 0.0),
            q_mean=cdiag.get("q_mean", 0.0),
            q_abs_max=cdiag.get("q_abs_max", 0.0),
            target_q_mean=cdiag.get("target_q_mean", 0.0),
            raw_actor_norm=adiag.get("raw_actor_norm", 0.0),
            bounded_actor_norm=adiag.get("bounded_actor_norm", 0.0),
            target_action_norm=cdiag.get("target_action_norm", 0.0),
            nan_inf=float(nan_inf),
        )
