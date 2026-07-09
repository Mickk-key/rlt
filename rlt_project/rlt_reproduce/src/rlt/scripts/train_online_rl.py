#!/usr/bin/env python3
"""Online RL on real FR3 (Franka Hand or Robotiq) or mock env (Algorithm 1)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from rich.console import Console

from rlt.hardware.gripper_factory import create_robot_env
from rlt.rl.learner import RLTLearner
from rlt.rl.online_loop import run_online_rl_episode
from rlt.rl.replay_buffer import ReplayBuffer
from rlt.rl_token.encoder_decoder import RLTokenEncoderDecoder, load_rl_token_state_dict
from rlt.sim.mock_env import MockPrecisionEnv
from rlt.vla.embedding_extractor import VLAEmbeddingExtractor

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/franka/fr3_franka.yaml"))
    parser.add_argument("--mock", action="store_true", help="Use mock env instead of robot")
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    rl = raw["online_rl"]
    rt = raw["rl_token"]
    vla_cfg = raw["vla"]
    device = raw.get("device", "cpu")
    episodes = args.episodes or rl.get("max_episodes", 50)

    token_dim = rt["token_dim"]
    proprio_dim = raw["robot"]["proprio_dim"]
    state_dim = token_dim + proprio_dim
    action_dim = rl["action_dim"]
    chunk_length = rl["chunk_length"]

    token_model = RLTokenEncoderDecoder(
        embed_dim=rt["embed_dim"],
        token_dim=token_dim,
        num_encoder_layers=rt["num_encoder_layers"],
        num_decoder_layers=rt["num_decoder_layers"],
        num_heads=rt["num_heads"],
        ff_dim=rt["ff_dim"],
        dropout=rt["dropout"],
    ).to(device)
    ckpt = Path(raw["paths"]["checkpoint_dir"]) / "rl_token.pt"
    if ckpt.exists():
        token_model.load_state_dict(
            load_rl_token_state_dict(ckpt, map_location=device, weights_only=True)
        )
        console.print(f"Loaded RL token from {ckpt}")

    vla = VLAEmbeddingExtractor(
        checkpoint=vla_cfg.get("checkpoint"),
        config_name=vla_cfg.get("config_name", "pi05_base"),
        device=device,
        action_dim=action_dim,
        chunk_horizon=vla_cfg.get("vla_chunk_horizon", 50),
        embed_dim=rt["embed_dim"],
    )

    if args.mock:
        env = MockPrecisionEnv(proprio_dim, action_dim, chunk_length)
        env.reset = lambda: np.zeros(proprio_dim, dtype=np.float32)  # type: ignore
        console.print("[yellow]Mock env mode[/yellow]")
    else:
        env = create_robot_env(raw)
        console.print("[green]Real robot env[/green]")

    buffer = ReplayBuffer(rl["replay_capacity"])
    learner = RLTLearner(
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_length=chunk_length,
        actor_hidden=rl["actor_hidden"],
        critic_hidden=rl["critic_hidden"],
        device=device,
        discount=rl["discount"],
        policy_constraint_beta=rl["policy_constraint_beta"],
        reference_dropout=rl["reference_dropout"],
        actor_lr=rl["actor_lr"],
        critic_lr=rl["critic_lr"],
        critic_ensemble=rl["critic_ensemble"],
        target_tau=rl["target_update_tau"],
    )

    language = vla_cfg.get("language_instruction", "")
    for ep in range(episodes):
        if args.mock:
            # wrap mock env for online_loop interface
            class MockWrap:
                def reset(self):
                    return np.zeros(proprio_dim, dtype=np.float32)

                def step(self, a):
                    _, r, d, _ = env.step(a)
                    return np.zeros(proprio_dim, dtype=np.float32), r, d, {}

            runner_env = MockWrap()
        else:
            runner_env = env

        stats = run_online_rl_episode(
            runner_env,
            learner,
            token_model,
            vla,
            buffer,
            chunk_length=chunk_length,
            execute_prefix=vla_cfg.get("execute_prefix", 20),
            control_hz=rl["control_hz"],
            language=language,
            update_ratio=rl["update_to_data_ratio"],
            batch_size=rl["batch_size"],
            critic_updates=rl["critic_updates_per_actor"],
            warmup=rl["warmup_steps"],
            device=device,
        )
        console.print(f"ep {ep}: {stats}")

    if not args.mock:
        env.close()

    actor_ckpt = Path(raw["paths"]["checkpoint_dir"]) / "rl_actor.pt"
    torch.save(learner.actor.state_dict(), actor_ckpt)
    console.print(f"[green]Saved actor[/green] {actor_ckpt}")


if __name__ == "__main__":
    main()
