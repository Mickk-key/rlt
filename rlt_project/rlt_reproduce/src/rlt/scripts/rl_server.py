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
import atexit
import json
import logging
import signal
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
from rlt.rl_token.encoder_decoder import RLTokenEncoderDecoder, load_rl_token_state_dict
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
            self.token_model.load_state_dict(
                load_rl_token_state_dict(token_ckpt, map_location=device, weights_only=True)
            )
            logger.info("Loaded RL token from %s", token_ckpt)

        self.vla = VLAEmbeddingExtractor(
            checkpoint=vla_cfg.get("checkpoint"),
            config_name=vla_cfg.get("config_name", "pi05_base"),
            device=device,
            action_dim=action_dim,
            chunk_horizon=vla_cfg.get("vla_chunk_horizon", 50),
            embed_dim=rt["embed_dim"],
            asset_id=vla_cfg.get("asset_id", "franka"),
            default_prompt=vla_cfg.get("default_prompt") or vla_cfg.get("language_instruction"),
            input_format=vla_cfg.get("input_format", "droid"),
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
            action_box=rl.get("action_box"),
            target_policy_noise=rl.get("target_policy_noise", 0.2),
            target_noise_clip=rl.get("target_noise_clip", 0.5),
            grad_clip_norm=rl.get("grad_clip_norm", 10.0),
        )

        actor_ckpt = ckpt_dir / "rl_actor.pt"
        critic_ckpt = ckpt_dir / "rl_critic.pt"
        if actor_ckpt.exists():
            self.learner.actor.load_state_dict(torch.load(actor_ckpt, map_location=device, weights_only=True))
            self.learner.actor_target.load_state_dict(self.learner.actor.state_dict())
            logger.info("Loaded actor from %s", actor_ckpt)
        if critic_ckpt.exists():
            self.learner.critic.load_state_dict(torch.load(critic_ckpt, map_location=device, weights_only=True))
            self.learner.critic_target.load_state_dict(self.learner.critic.state_dict())
            logger.info("Loaded critic from %s", critic_ckpt)

        self._ckpt_dir = ckpt_dir
        self._online_ckpt_dir = ckpt_dir / "online_rl"
        self._train_steps = 0
        self._checkpoint_save_interval = int(rl.get("checkpoint_save_interval_updates", 50))
        self._checkpoint_keep_last = int(rl.get("checkpoint_keep_last", 5))
        self._checkpoint_save_on_shutdown = bool(rl.get("checkpoint_save_on_shutdown", True))

        self._warmup = rl["warmup_steps"]
        self._batch_size = rl["batch_size"]
        self._update_ratio = rl["update_to_data_ratio"]
        self._critic_updates = rl["critic_updates_per_actor"]

        infer_cfg = raw.get("inference", {})
        # "auto" = paper-faithful warmup-gated execution (Algorithm 1 line 9).
        # "reference" / "reference_noise" / "policy" = manual debug overrides.
        self.inference_mode = infer_cfg.get("mode", "auto")
        self._manual_override = None if self.inference_mode == "auto" else self.inference_mode
        self._policy_ramp_steps = int(infer_cfg.get("ramp_steps", 0))
        anchor_cfg = raw.get("policy_anchor", {})
        self.inference_policy = RLTInferencePolicy(
            self.learner,
            noise_std=float(infer_cfg.get("reference_noise_std", rl.get("action_noise_std", 0.05))),
            anchor_enabled=bool(anchor_cfg.get("enabled", True)),
            max_dev_trans_m=float(anchor_cfg.get("max_dev_trans_m", 0.01)),
            max_dev_rot_rad=float(anchor_cfg.get("max_dev_rot_rad", 0.05)),
            max_dev_grip=float(anchor_cfg.get("max_dev_grip", 1.0)),
            action_dim=action_dim,
        )
        self._image_size = tuple(raw.get("cameras", {}).get("image_size", [224, 224]))
        logger.info("Inference mode: %s", self.inference_mode)

    def _atomic_torch_save(self, obj: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp)
        tmp.replace(path)

    def save_checkpoints(self, *, reason: str = "interval") -> dict[str, str]:
        """Persist actor/critic for resume; keep a few timestamped snapshots."""
        actor_path = self._ckpt_dir / "rl_actor.pt"
        critic_path = self._ckpt_dir / "rl_critic.pt"
        self._atomic_torch_save(self.learner.actor.state_dict(), actor_path)
        self._atomic_torch_save(self.learner.critic.state_dict(), critic_path)

        saved = {
            "actor": str(actor_path),
            "critic": str(critic_path),
            "reason": reason,
            "train_steps": str(self._train_steps),
            "buffer_size": str(len(self.buffer)),
        }

        if self._checkpoint_keep_last > 0:
            snap_dir = self._online_ckpt_dir
            snap_dir.mkdir(parents=True, exist_ok=True)
            tag = f"step{self._train_steps:06d}_buf{len(self.buffer)}"
            snap_actor = snap_dir / f"rl_actor_{tag}.pt"
            snap_critic = snap_dir / f"rl_critic_{tag}.pt"
            self._atomic_torch_save(self.learner.actor.state_dict(), snap_actor)
            self._atomic_torch_save(self.learner.critic.state_dict(), snap_critic)
            saved["snapshot_actor"] = str(snap_actor)
            actors = sorted(snap_dir.glob("rl_actor_step*.pt"))
            critics = sorted(snap_dir.glob("rl_critic_step*.pt"))
            for stale in actors[: max(0, len(actors) - self._checkpoint_keep_last)]:
                stale.unlink(missing_ok=True)
            for stale in critics[: max(0, len(critics) - self._checkpoint_keep_last)]:
                stale.unlink(missing_ok=True)

        logger.info(
            "Saved checkpoints (%s): actor=%s critic=%s train_steps=%d buffer=%d",
            reason,
            actor_path,
            critic_path,
            self._train_steps,
            len(self.buffer),
        )
        return saved

    def shutdown(self) -> None:
        if self._checkpoint_save_on_shutdown and self._train_steps > 0:
            self.save_checkpoints(reason="shutdown")

    def _maybe_save_checkpoints(self) -> None:
        interval = self._checkpoint_save_interval
        if interval <= 0 or self._train_steps <= 0:
            return
        if self._train_steps % interval == 0:
            self.save_checkpoints(reason="interval")

    def _as_state_vector(self, arr: Any, *, label: str) -> np.ndarray:
        s = np.asarray(arr, dtype=np.float32).reshape(-1)
        if s.shape[0] != self._state_dim:
            raise ValueError(
                f"{label} dim {s.shape[0]} != expected state_dim {self._state_dim} "
                f"(token_dim + proprio_dim)"
            )
        return s

    def _as_action_chunk(self, arr: Any, *, label: str) -> np.ndarray:
        """Validate a REAL action chunk of shape ``(chunk_length, action_dim)``.

        Phase 4: single-step tiling (``np.tile(a, (C, 1))``) is removed — the robot
        must send genuine executed/reference chunks so the critic sees ``a_{t:t+C}``
        and the temporal gap matches ``gamma^C``. Only genuine (C, action_dim)
        arrays are accepted; anything else is a bug and must fail loudly.
        """
        a = np.asarray(arr, dtype=np.float32)
        cl, ad = self.chunk_length, self._action_dim
        if a.shape == (cl, ad):
            return a
        raise ValueError(
            f"{label} shape {a.shape} != required real chunk ({cl}, {ad}); "
            f"single-step tiling is not allowed (Phase 4)"
        )

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

    def _state_from_vla_output(
        self,
        vla_out,
        proprio: np.ndarray,
    ) -> np.ndarray:
        with torch.no_grad():
            emb = torch.as_tensor(vla_out.embeddings, device=self.device).unsqueeze(0)
            z = self.token_model.encode(emb).squeeze(0).cpu().numpy()
        return np.concatenate([z, proprio.astype(np.float32)])

    def _encode_state(
        self,
        proprio: np.ndarray,
        images: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        vla_out = self.vla.infer_from_proprio(proprio, images=images, language=self.language)
        return self._state_from_vla_output(vla_out, proprio)

    def handle_infer(self, msg: dict[str, Any]) -> dict[str, Any]:
        import time

        proprio, images = self._vla_inputs(msg)
        t0 = time.perf_counter()
        vla_out = self.vla.infer_from_proprio(proprio, images=images, language=self.language)
        vla_sec = time.perf_counter() - t0
        ref = vla_out.reference_action[: self.chunk_length]
        state = self._state_from_vla_output(vla_out, proprio)
        action_np, meta = self.inference_policy.act_gated(
            state,
            ref,
            buffer_size=len(self.buffer),
            warmup_steps=self._warmup,
            ramp_steps=self._policy_ramp_steps,
            override=self._manual_override,
        )
        logger.info(
            "[exec] mode=%s override=%s buffer=%d/%d ramp=%d alpha=%.3f "
            "ref_norm=%.4f policy_norm=%s executed_norm=%.4f anchor_clipped=%s",
            meta.get("exec_mode"),
            self._manual_override,
            len(self.buffer),
            self._warmup,
            self._policy_ramp_steps,
            float(meta.get("alpha", 0.0)),
            float(meta.get("ref_norm", 0.0)),
            f"{meta['policy_norm']:.4f}" if "policy_norm" in meta else "n/a",
            float(meta.get("executed_norm", 0.0)),
            meta.get("anchor_clipped", "n/a"),
        )
        total_sec = time.perf_counter() - t0
        if vla_sec > 5.0 or total_sec > 5.0:
            logger.info("infer timing: vla=%.2fs total=%.2fs", vla_sec, total_sec)
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

    def _aggregate_chunk_reward(self, msg: dict[str, Any]) -> float:
        """In-chunk discounted return R = sum_{k=0}^{C-1} gamma^k r_{t+k} (Eq. 3).

        Robot sends per-step ``rewards`` for the chunk; discounting uses the learner's
        gamma so the exponent convention stays in one place. A pre-aggregated scalar
        ``reward`` is accepted as a fallback (legacy / single-env path).
        """
        if "rewards" in msg:
            r = np.asarray(msg["rewards"], dtype=np.float32).reshape(-1)
            k = np.arange(r.shape[0], dtype=np.float32)
            return float(np.sum((self.learner.discount**k) * r))
        return float(msg["reward"])

    def handle_encode(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Encode an observation into the RL state x = [z_rl | proprio].

        Used by the robot to build the per-step state stream (x_0..x_T) so it can
        assemble paper-faithful chunk transitions with the correct C-step gap.
        """
        proprio, images = self._vla_inputs(msg)
        state = self._encode_state(proprio, images)
        return {"type": "encode_response", "state": state.tolist()}

    def handle_transition(self, msg: dict[str, Any]) -> dict[str, Any]:
        state = self._as_state_vector(msg["state"], label="state")
        next_state = self._as_state_vector(self._resolve_next_state(msg), label="next_state")
        action = self._as_action_chunk(msg["action"], label="action")
        reference = self._as_action_chunk(msg["reference_action"], label="reference_action")
        next_ref_raw = msg.get("next_reference_action", msg["reference_action"])
        next_reference = self._as_action_chunk(next_ref_raw, label="next_reference_action")
        reward = self._aggregate_chunk_reward(msg)
        self.buffer.add(
            Transition(
                state=state,
                action=action,
                reference_action=reference,
                reward=reward,
                next_state=next_state,
                done=bool(msg["done"]),
                next_reference_action=next_reference,
                intervened=float(msg.get("intervened", 0.0)),
            )
        )
        updated = False
        metrics = None
        if len(self.buffer) >= self._warmup:
            for _ in range(self._update_ratio):
                if len(self.buffer) >= self._batch_size:
                    metrics = self.learner.train_step(self.buffer, self._batch_size, self._critic_updates)
                    self._train_steps += 1
                    updated = True
                    self._maybe_save_checkpoints()
        resp: dict[str, Any] = {
            "type": "transition_response",
            "buffer_size": len(self.buffer),
            "updated": updated,
            "train_steps": self._train_steps,
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
                buf = len(self.buffer)
                if self._manual_override is not None:
                    exec_state = f"override:{self._manual_override}"
                elif buf < self._warmup:
                    exec_state = "warmup_reference"
                elif self._policy_ramp_steps > 0 and buf < self._warmup + self._policy_ramp_steps:
                    exec_state = "ramp"
                else:
                    exec_state = "policy"
                resp = {
                    "type": "pong",
                    "buffer_size": buf,
                    "device": self.device,
                    "warmup_steps": self._warmup,
                    "ramp_steps": self._policy_ramp_steps,
                    "training": buf >= self._warmup,
                    "train_steps": self._train_steps,
                    "inference_mode": self.inference_mode,
                    "exec_state": exec_state,
                }
            elif msg_type == "infer":
                resp = self.handle_infer(msg)
            elif msg_type == "encode":
                resp = self.handle_encode(msg)
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

    async with websockets.serve(
        handler,
        host,
        port,
        # The VLA infer inside handle_message runs synchronously and blocks the asyncio
        # event loop for 1-2 min. With default keepalive (ping every 20s, 20s timeout)
        # the connection is force-closed mid-infer ("1011 keepalive ping timeout").
        # Disable keepalive; the client uses an explicit per-request recv timeout instead.
        ping_interval=None,
        ping_timeout=None,
        max_size=None,  # allow large JPEG image payloads
        close_timeout=5,
    ):
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
    device = args.device or raw.get("device", "cuda")

    server = RLServer(raw, device=device)
    atexit.register(server.shutdown)

    def _handle_signal(signum, _frame):
        logger.info("Received signal %s — saving checkpoints and exiting", signum)
        server.shutdown()
        raise SystemExit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    asyncio.run(_serve(server, args.host, args.port))


if __name__ == "__main__":
    main()
