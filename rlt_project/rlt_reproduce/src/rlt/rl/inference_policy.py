"""GPU-side action selection for online RL rollout."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

InferenceMode = Literal["reference", "reference_noise", "policy"]


class RLTInferencePolicy:
    """Choose executed action from VLA reference and/or RL actor."""

    def __init__(self, learner, *, noise_std: float = 0.05, deterministic: bool = True) -> None:
        self.learner = learner
        self.noise_std = float(noise_std)
        self.deterministic = deterministic

    def act(
        self,
        state: np.ndarray,
        reference: np.ndarray,
        *,
        mode: InferenceMode = "reference",
    ) -> tuple[np.ndarray, dict[str, float | str]]:
        ref = np.asarray(reference, dtype=np.float32)
        meta: dict[str, float | str] = {
            "policy_mode": mode,
            "ref_norm": float(np.linalg.norm(ref)),
        }

        if mode == "reference":
            action = ref.copy()
            meta["z_rl_norm"] = float(np.linalg.norm(state[: max(1, len(state) - 8)]))
            return action, meta

        if mode == "reference_noise":
            noise = np.random.randn(*ref.shape).astype(np.float32) * self.noise_std
            action = ref + noise
            meta["noise_std"] = self.noise_std
            meta["z_rl_norm"] = float(np.linalg.norm(state[: max(1, len(state) - 8)]))
            return action, meta

        ref_t = torch.as_tensor(ref, dtype=torch.float32, device=self.learner.device).unsqueeze(0)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.learner.device).unsqueeze(0)
        with torch.no_grad():
            if self.deterministic:
                action_t, _ = self.learner.actor(state_t, ref_t)
            else:
                action_t, _ = self.learner.actor.sample(state_t, ref_t)
        action = action_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
        meta["z_rl_norm"] = float(np.linalg.norm(state[: max(1, len(state) - 8)]))
        meta["policy_norm"] = float(np.linalg.norm(action))
        return action, meta
