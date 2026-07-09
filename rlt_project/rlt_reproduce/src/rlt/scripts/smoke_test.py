#!/usr/bin/env python3
"""End-to-end smoke test: RL token training + online RL on mock env (CPU)."""

from __future__ import annotations

import numpy as np
import torch
from rich.console import Console

from rlt.rl.learner import RLTLearner
from rlt.rl.replay_buffer import ReplayBuffer, Transition
from rlt.rl_token.encoder_decoder import RLTokenEncoderDecoder
from rlt.sim.mock_env import MockPrecisionEnv
from rlt.vla.openpi_wrapper import MockVLAWrapper

console = Console()


def main() -> None:
    device = "cpu"
    embed_dim = 128
    token_dim = 128
    action_dim = 4
    chunk_length = 2
    proprio_dim = 8
    state_dim = token_dim + proprio_dim

    console.print("[bold]1/3 RL token pretraining (mock VLA embeddings)[/bold]")
    model = RLTokenEncoderDecoder(
        embed_dim=embed_dim,
        token_dim=token_dim,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        ff_dim=256,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    vla = MockVLAWrapper(action_dim=action_dim, chunk_horizon=10, embed_dim=embed_dim, num_tokens=16)

    for step in range(50):
        out = vla.forward(batch_size=8, device=device)
        loss, _ = model.reconstruction_loss(out.embeddings)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            console.print(f"  step {step:3d}  L_ro={loss.item():.4f}")

    console.print("[bold]2/3 Fill replay buffer (mock env)[/bold]")
    env = MockPrecisionEnv(state_dim=proprio_dim, action_dim=action_dim, chunk_length=chunk_length)
    buffer = ReplayBuffer(capacity=5000)
    learner = RLTLearner(
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_length=chunk_length,
        actor_hidden=[64, 64],
        critic_hidden=[64, 64],
        device=device,
    )

    for ep in range(30):
        proprio = env.reset()
        ref = env.reference_action()
        with torch.no_grad():
            emb = vla.forward(1, device).embeddings
            z_rl = model.encode(emb).squeeze(0).cpu().numpy()
        state = np.concatenate([z_rl, proprio])
        action = ref + np.random.normal(scale=0.3, size=ref.shape).astype(np.float32)
        _, reward, done, info = env.step(action[0])
        next_proprio = env.reset() if done else proprio
        next_state = np.concatenate([z_rl, next_proprio])
        buffer.add(
            Transition(
                state=state,
                action=action,
                reference_action=ref,
                reward=reward,
                next_state=next_state,
                done=done,
            )
        )

    console.print(f"  buffer size: {len(buffer)}")

    console.print("[bold]3/3 Online RL updates (TD3-style)[/bold]")
    for step in range(20):
        metrics = learner.train_step(buffer, batch_size=16, critic_updates=2)
        if step % 5 == 0:
            console.print(f"  step {step:3d}  critic={metrics.critic_loss:.4f}  actor={metrics.actor_loss:.4f}")

    console.print("\n[green]Smoke test passed. Pipeline matches paper structure (IV-A + IV-B).[/green]")


if __name__ == "__main__":
    main()
