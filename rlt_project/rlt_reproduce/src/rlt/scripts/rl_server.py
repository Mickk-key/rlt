#!/usr/bin/env python3
"""GPU-side RL server for split online RL (runs on the training host).

Handles:
  infer     — VLA reference + RL actor action chunk
  transition — replay buffer write + TD3 learner updates

Start on GPU host:
  python -m rlt.scripts.rl_server --config configs/franka/fr3_franka.yaml --device cuda

Robot PC connects via ``gpu_client.WebsocketRLClient`` (see ONLINE_RL_TASKS.md).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
import websockets

from rlt.rl.gpu_client import _json_default
from rlt.rl.inference_policy import RLTInferencePolicy
from rlt.rl.learner import RLTLearner
from rlt.rl.replay_buffer import ReplayBuffer, Transition
from rlt.rl.ws_protocol import decode_images_jpeg, resize_rgb_frames
from rlt.rl_token.encoder_decoder import RLTokenEncoderDecoder
from rlt.util.deoxys_paths import apply_deoxys_paths
from rlt.vla.embedding_extractor import VLAEmbeddingExtractor

logger = logging.getLogger(__name__)


class RLServer:
    def __init__(self, raw: dict, *, device: str = "cuda") -> None:
        self.device = device
        rl = raw["online_rl"]
        rt = raw["rl_token"]
        vla_cfg = raw["vla"]
        proprio_dim = raw["robot"]["proprio_dim"]
        token_dim = rt["token_dim"]
        state_dim = token_dim + proprio_dim
        action_dim = rl["action_dim"]
        chunk_length = rl["chunk_length"]

        self.chunk_length = chunk_length
        self._action_dim = action_dim
        self._state_dim = state_dim
        self.language = vla_cfg.get("language_instruction", "")

        self.token_model = RLTokenEncoderDecoder(
            embed_dim=rt["embed_dim"],
            token_dim=token_dim,
            num_encoder_layers=rt["num_encoder_layers"],
            num_decoder_layers=rt["num_decoder_layers"],
            num_heads=rt["num_heads"],
            ff_dim=rt["ff_dim"],
            dropout=rt["dropout"],
        ).to(device)

        ckpt_dir = Path(raw["paths"]["checkpoint_dir"])
        token_ckpt = ckpt_dir / "rl_token.pt"
        if token_ckpt.exists():
            self.token_model.load_state_dict(torch.load(token_ckpt, map_location=device, weights_only=True))
            logger.info("Loaded RL token from %s", token_ckpt)

        self.vla = VLAEmbeddingExtractor(
            checkpoint=vla_cfg.get("checkpoint"),
            config_name=vla_cfg.get("config_name", "pi05_base"),
            device=device,
            action_dim=action_dim,
            chunk_horizon=vla_cfg.get("vla_chunk_horizon", 50),
            embed_dim=rt["embed_dim"],
        )

        self.buffer = ReplayBuffer(rl["replay_capacity"])
        self.learner = RLTLearner(
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

        actor_ckpt = ckpt_dir / "rl_actor.pt"
        if actor_ckpt.exists():
            self.learner.actor.load_state_dict(torch.load(actor_ckpt, map_location=device, weights_only=True))
            logger.info("Loaded actor from %s", actor_ckpt)

        self._warmup = rl["warmup_steps"]
        self._batch_size = rl["batch_size"]
        self._update_ratio = rl["update_to_data_ratio"]
        self._critic_updates = rl["critic_updates_per_actor"]

        infer_cfg = raw.get("inference", {})
        self.inference_mode = infer_cfg.get("mode", "reference")
        self.inference_policy = RLTInferencePolicy(
            self.learner,
            noise_std=float(infer_cfg.get("reference_noise_std", rl.get("action_noise_std", 0.05))),
        )
        self._image_size = tuple(raw.get("cameras", {}).get("image_size", [224, 224]))
        logger.info("Inference mode: %s", self.inference_mode)

    def _as_state_vector(self, arr: Any, *, label: str) -> np.ndarray:
        s = np.asarray(arr, dtype=np.float32).reshape(-1)
        if s.shape[0] != self._state_dim:
            raise ValueError(
                f"{label} dim {s.shape[0]} != expected state_dim {self._state_dim} "
                f"(token_dim + proprio_dim)"
            )
        return s

    def _as_action_chunk(self, arr: Any, *, label: str) -> np.ndarray:
        """Expand per-step robot action (7,) to learner chunk (chunk_length, action_dim)."""
        a = np.asarray(arr, dtype=np.float32)
        cl, ad = self.chunk_length, self._action_dim
        if a.ndim == 1:
            if a.shape[0] != ad:
                raise ValueError(f"{label} dim {a.shape[0]} != action_dim {ad}")
            return np.tile(a, (cl, 1))
        if a.ndim == 2 and a.shape[1] == ad:
            if a.shape[0] == cl:
                return a
            if a.shape[0] == 1:
                return np.tile(a, (cl, 1))
            if a.shape[0] < cl:
                pad = np.repeat(a[-1:], cl - a.shape[0], axis=0)
                return np.concatenate([a, pad], axis=0)
            return a[:cl]
        raise ValueError(f"{label} shape {a.shape} invalid; want ({ad},) or ({cl}, {ad})")

    def _vla_inputs(self, msg: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
        proprio = np.asarray(msg["proprio"], dtype=np.float32)
        images = None
        if "images_jpeg" in msg:
            images = decode_images_jpeg(msg["images_jpeg"])
        elif "images" in msg:
            images = {k: np.asarray(v) for k, v in msg["images"].items()}
        if images:
            logger.info(
                "Received images: %s",
                {k: tuple(np.asarray(v).shape) for k, v in images.items()},
            )
            if self._image_size:
                images = resize_rgb_frames(images, self._image_size)
        else:
            logger.warning("infer/transition message has NO images (images_jpeg missing or empty)")
        return proprio, images

    def _encode_state(
        self,
        proprio: np.ndarray,
        images: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        with torch.no_grad():
            vla_out = self.vla.infer_from_proprio(proprio, images=images, language=self.language)
            emb = torch.as_tensor(vla_out.embeddings, device=self.device).unsqueeze(0)
            z = self.token_model.encode(emb).squeeze(0).cpu().numpy()
        return np.concatenate([z, proprio.astype(np.float32)])

    def handle_infer(self, msg: dict[str, Any]) -> dict[str, Any]:
        proprio, images = self._vla_inputs(msg)
        state = self._encode_state(proprio, images)
        with torch.no_grad():
            vla_out = self.vla.infer_from_proprio(proprio, images=images, language=self.language)
            ref = vla_out.reference_action[: self.chunk_length]
        action_np, meta = self.inference_policy.act(
            state,
            ref,
            mode=self.inference_mode,
        )
        return {
            "type": "infer_response",
            "action_chunk": action_np.tolist(),
            "reference_action": ref.tolist(),
            "state": state.tolist(),
            **meta,
        }

    def _resolve_next_state(self, msg: dict[str, Any]) -> np.ndarray:
        if "next_state" in msg:
            return np.asarray(msg["next_state"], dtype=np.float32)
        if "next_proprio" not in msg:
            raise KeyError("transition requires next_state or next_proprio")
        next_proprio = np.asarray(msg["next_proprio"], dtype=np.float32)
        images = None
        if "next_images_jpeg" in msg:
            images = decode_images_jpeg(msg["next_images_jpeg"])
            if self._image_size:
                images = resize_rgb_frames(images, self._image_size)
        elif "next_images" in msg:
            images = {k: np.asarray(v) for k, v in msg["next_images"].items()}
            if self._image_size:
                images = resize_rgb_frames(images, self._image_size)
        if images is None:
            raise ValueError(
                "transition missing next_images_jpeg — robot must send wrist + external "
                "frames for VLA next_state encoding (observation/wrist_image_left)"
            )
        return self._encode_state(next_proprio, images)

    def handle_transition(self, msg: dict[str, Any]) -> dict[str, Any]:
        state = self._as_state_vector(msg["state"], label="state")
        next_state = self._as_state_vector(self._resolve_next_state(msg), label="next_state")
        action = self._as_action_chunk(msg["action"], label="action")
        reference = self._as_action_chunk(msg["reference_action"], label="reference_action")
        self.buffer.add(
            Transition(
                state=state,
                action=action,
                reference_action=reference,
                reward=float(msg["reward"]),
                next_state=next_state,
                done=bool(msg["done"]),
            )
        )
        updated = False
        metrics = None
        if len(self.buffer) >= self._warmup:
            for _ in range(self._update_ratio):
                if len(self.buffer) >= self._batch_size:
                    metrics = self.learner.train_step(self.buffer, self._batch_size, self._critic_updates)
                    updated = True
        resp: dict[str, Any] = {
            "type": "transition_response",
            "buffer_size": len(self.buffer),
            "updated": updated,
            "next_state": next_state.tolist(),
        }
        if metrics is not None:
            resp["metrics"] = asdict(metrics)
        return resp

    async def handle_message(self, raw_msg: str) -> str:
        try:
            msg = json.loads(raw_msg)
            msg_type = msg.get("type")
            if msg_type == "ping":
                resp = {
                    "type": "pong",
                    "buffer_size": len(self.buffer),
                    "device": self.device,
                    "warmup_steps": self._warmup,
                    "training": len(self.buffer) >= self._warmup,
                    "inference_mode": self.inference_mode,
                }
            elif msg_type == "infer":
                resp = self.handle_infer(msg)
            elif msg_type == "transition":
                resp = self.handle_transition(msg)
            else:
                resp = {"type": "error", "message": f"unknown type: {msg_type}"}
        except Exception as exc:
            logger.exception("RL server message failed")
            resp = {"type": "error", "message": str(exc)}
        return json.dumps(resp, default=_json_default)


async def _serve(server: RLServer, host: str, port: int) -> None:
    async def handler(ws):
        async for raw in ws:
            reply = await server.handle_message(raw)
            await ws.send(reply)

    async with websockets.serve(handler, host, port):
        logger.info("RL server listening on ws://%s:%d", host, port)
        await asyncio.Future()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/franka/fr3_franka.yaml"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    apply_deoxys_paths(raw, smq_root=Path(__file__).resolve().parents[4])
    device = args.device or raw.get("device", "cuda")

    server = RLServer(raw, device=device)
    asyncio.run(_serve(server, args.host, args.port))


if __name__ == "__main__":
    main()
