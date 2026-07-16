"""GPU-side action selection for online RL rollout."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

InferenceMode = Literal["reference", "reference_noise", "policy"]


class RLTInferencePolicy:
    """Choose executed action from VLA reference and/or RL actor.

    The actor keeps the paper's parameterization (Eq. 4): it outputs an *absolute*
    Gaussian action chunk conditioned on the VLA reference, ``a ~ pi(a | x, ref)`` —
    NOT a residual network. For safe deployment we additionally anchor the executed
    action to the reference by bounding the deviation:

        action = reference + clip(actor_action - reference, +/- delta)

    This preserves the paper semantics ("online RL locally edits the VLA action")
    while guaranteeing the executed chunk never strays arbitrarily far from the
    known-safe VLA proposal. It is complementary to (and upstream of) the hard
    per-step robot safety clamp in ``DeoxysEnv.step``.
    """

    def __init__(
        self,
        learner,
        *,
        noise_std: float = 0.05,
        deterministic: bool = True,
        anchor_enabled: bool = True,
        max_dev_trans_m: float = 0.01,
        max_dev_rot_rad: float = 0.05,
        max_dev_grip: float = 1.0,
        action_dim: int = 7,
    ) -> None:
        self.learner = learner
        self.noise_std = float(noise_std)
        self.deterministic = deterministic
        self.anchor_enabled = bool(anchor_enabled)
        self.max_dev_trans_m = float(max_dev_trans_m)
        self.max_dev_rot_rad = float(max_dev_rot_rad)
        self.max_dev_grip = float(max_dev_grip)
        self.action_dim = int(action_dim)

    @staticmethod
    def _clip_rows_norm(vecs: np.ndarray, cap: float) -> tuple[np.ndarray, bool]:
        """Scale each row down to ``cap`` magnitude, preserving direction."""
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        over = norms > cap
        if not np.any(over):
            return vecs, False
        scale = np.ones_like(norms)
        safe = np.maximum(norms, 1e-12)
        scale = np.where(over, cap / safe, 1.0)
        return (vecs * scale).astype(vecs.dtype), True

    def _anchor_to_reference(
        self,
        action: np.ndarray,
        ref: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Bound ``action`` to a local edit of ``ref`` (deviation clip)."""
        act2 = np.atleast_2d(action).astype(np.float32)
        ref2 = np.atleast_2d(ref).astype(np.float32)
        dev = act2 - ref2
        raw_dev_norm = float(np.linalg.norm(dev))

        info: dict[str, float] = {"actor_ref_dist": raw_dev_norm}

        if act2.shape[-1] >= 6:
            info["max_step_trans_dev_m"] = float(np.linalg.norm(dev[:, :3], axis=-1).max())
            info["max_step_rot_dev_rad"] = float(np.linalg.norm(dev[:, 3:6], axis=-1).max())

        if not self.anchor_enabled:
            info["anchor_clipped"] = 0.0
            info["clipped_dev_norm"] = raw_dev_norm
            return action, info

        clipped = False
        if act2.shape[-1] >= 7:
            dev[:, :3], t_clip = self._clip_rows_norm(dev[:, :3], self.max_dev_trans_m)
            dev[:, 3:6], r_clip = self._clip_rows_norm(dev[:, 3:6], self.max_dev_rot_rad)
            g = dev[:, 6:7]
            g_clipped = np.clip(g, -self.max_dev_grip, self.max_dev_grip)
            grip_clip = not np.allclose(g, g_clipped, atol=1e-6)
            dev[:, 6:7] = g_clipped
            clipped = bool(t_clip or r_clip or grip_clip)
        else:
            # Generic fallback: bound the whole deviation vector magnitude.
            dev, clipped = self._clip_rows_norm(dev, self.max_dev_trans_m)

        anchored = (ref2 + dev).astype(np.float32).reshape(action.shape)
        info["anchor_clipped"] = float(bool(clipped))
        info["clipped_dev_norm"] = float(np.linalg.norm(dev))
        return anchored, info

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
        actor_action = action_t.squeeze(0).detach().cpu().numpy().astype(np.float32)

        # Raw actor output (before anchoring) — key diagnostic for OOD detection.
        meta["policy_norm"] = float(np.linalg.norm(actor_action))
        meta["z_rl_norm"] = float(np.linalg.norm(state[: max(1, len(state) - 8)]))

        action, anchor_info = self._anchor_to_reference(actor_action, ref)
        meta.update({k: float(v) for k, v in anchor_info.items()})
        meta["anchored_norm"] = float(np.linalg.norm(action))
        return action, meta

    def act_gated(
        self,
        state: np.ndarray,
        reference: np.ndarray,
        *,
        buffer_size: int,
        warmup_steps: int,
        ramp_steps: int = 0,
        override: str | None = None,
    ) -> tuple[np.ndarray, dict[str, float | str]]:
        """Warmup-gated execution (RLT Algorithm 1, line 9).

        Chooses the executed action purely from the replay-buffer *transition count*:

            buffer_size <  N_warm                 -> reference only          (alpha = 0)
            N_warm <= buffer_size < N_warm+ramp   -> ramp reference->policy  (0 < alpha < 1)
            buffer_size >= N_warm + ramp           -> anchored policy         (alpha = 1)

        The ramp linearly interpolates the executed chunk to avoid a sudden 100%
        reference -> 100% actor switch:

            executed = (1 - alpha) * reference + alpha * anchored_policy

        ``override`` bypasses gating for debugging: "reference", "reference_noise",
        or "policy" (force actor, ignoring warmup). ``None`` = paper-faithful auto.
        """
        ref = np.asarray(reference, dtype=np.float32)
        buffer_size = int(buffer_size)
        warmup_steps = int(warmup_steps)
        ramp_steps = int(max(0, ramp_steps))

        if override in ("reference", "reference_noise"):
            alpha, label = 0.0, override
        elif override == "policy":
            alpha, label = 1.0, "policy(override)"
        elif buffer_size < warmup_steps:
            alpha, label = 0.0, "warmup_reference"
        elif ramp_steps > 0 and buffer_size < warmup_steps + ramp_steps:
            alpha = float(np.clip((buffer_size - warmup_steps) / ramp_steps, 0.0, 1.0))
            label = "ramp"
        else:
            alpha, label = 1.0, "policy"

        meta: dict[str, float | str] = {
            "exec_mode": label,
            "buffer_size": float(buffer_size),
            "warmup_steps": float(warmup_steps),
            "ramp_steps": float(ramp_steps),
            "alpha": float(alpha),
            "ref_norm": float(np.linalg.norm(ref)),
            "z_rl_norm": float(np.linalg.norm(state[: max(1, len(state) - 8)])),
        }

        if override == "reference_noise":
            noise = np.random.randn(*ref.shape).astype(np.float32) * self.noise_std
            executed = ref + noise
            meta["policy_mode"] = "reference_noise"
            meta["noise_std"] = self.noise_std
            meta["executed_norm"] = float(np.linalg.norm(executed))
            return executed, meta

        if alpha <= 0.0:
            executed = ref.copy()
            meta["policy_mode"] = "reference"
            meta["executed_norm"] = float(np.linalg.norm(executed))
            return executed, meta

        policy_action, pmeta = self.act(state, ref, mode="policy")
        for k in (
            "policy_norm",
            "actor_ref_dist",
            "anchor_clipped",
            "clipped_dev_norm",
            "anchored_norm",
            "max_step_trans_dev_m",
            "max_step_rot_dev_rad",
        ):
            if k in pmeta:
                meta[k] = float(pmeta[k])

        executed = ((1.0 - alpha) * ref + alpha * policy_action).astype(np.float32)
        # policy_mode != "reference" tells the robot to execute action_chunk (the blend).
        meta["policy_mode"] = "policy" if alpha >= 1.0 else "ramp"
        meta["executed_norm"] = float(np.linalg.norm(executed))
        return executed, meta
