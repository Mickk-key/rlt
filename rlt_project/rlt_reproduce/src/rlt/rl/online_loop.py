"""Shared online RL loop (Algorithm 1) for mock or real env."""

from __future__ import annotations

import time

import numpy as np
import torch

from rlt.rl.learner import RLTLearner
from rlt.rl.replay_buffer import ReplayBuffer, Transition
from rlt.rl_token.encoder_decoder import RLTokenEncoderDecoder
from rlt.vla.embedding_extractor import VLAEmbeddingExtractor


def run_online_rl_episode(
    env,
    learner: RLTLearner,
    token_model: RLTokenEncoderDecoder,
    vla: VLAEmbeddingExtractor,
    buffer: ReplayBuffer,
    *,
    chunk_length: int,
    execute_prefix: int,
    control_hz: float,
    language: str = "",
    reward_fn=None,
    max_steps: int = 200,
    update_ratio: int = 5,
    batch_size: int = 256,
    critic_updates: int = 2,
    warmup: int = 500,
    use_rl_policy: bool = True,
    device: str = "cpu",
) -> dict:
    """One critical-phase episode with chunk-level TD3 updates."""
    dt = 1.0 / control_hz
    torch_device = torch.device(device)
    proprio = env.reset()
    total_reward = 0.0
    steps = 0
    transitions = 0

    while steps < max_steps:
        with torch.no_grad():
            vla_out = vla.infer_from_proprio(proprio, language=language)
            ref = vla_out.reference_action[:chunk_length]
            emb = torch.as_tensor(vla_out.embeddings, device=torch_device).unsqueeze(0)
            z = token_model.encode(emb).squeeze(0).cpu().numpy()
        state = np.concatenate([z, proprio])

        ref_t = torch.as_tensor(ref, dtype=torch.float32, device=learner.device).unsqueeze(0)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=learner.device).unsqueeze(0)
        if use_rl_policy:
            action, _ = learner.actor.sample(state_t, ref_t)
            action_np = action.squeeze(0).detach().cpu().numpy()
        else:
            action_np = ref.copy()

        for i in range(min(chunk_length, execute_prefix)):
            a = action_np[i] if action_np.ndim == 2 else action_np
            next_proprio, env_reward, done, info = env.step(a)
            reward = reward_fn(proprio, a, next_proprio) if reward_fn else env_reward
            with torch.no_grad():
                vla_next = vla.infer_from_proprio(next_proprio, language=language)
                emb_n = torch.as_tensor(vla_next.embeddings, device=torch_device).unsqueeze(0)
                z_n = token_model.encode(emb_n).squeeze(0).cpu().numpy()
            next_state = np.concatenate([z_n, next_proprio])
            buffer.add(
                Transition(
                    state=state,
                    action=action_np if action_np.ndim == 2 else action_np[None, :],
                    reference_action=ref,
                    reward=float(reward),
                    next_state=next_state,
                    done=bool(done),
                )
            )
            transitions += 1
            total_reward += float(reward)
            proprio = next_proprio
            state = next_state
            steps += 1
            if done:
                break
            time.sleep(dt)
        if done:
            break
        if len(buffer) >= warmup:
            for _ in range(update_ratio):
                if len(buffer) >= batch_size:
                    learner.train_step(buffer, batch_size, critic_updates)

    return {
        "steps": steps,
        "transitions": transitions,
        "total_reward": total_reward,
        "buffer_size": len(buffer),
    }
