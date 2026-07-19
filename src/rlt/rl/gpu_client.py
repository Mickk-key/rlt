"""GPU-side inference client for split online RL (robot PC ↔ GPU host).

RUNTIME AUTHORITATIVE COPY: this ``smq&jgy/src`` copy is first on the robot
``PYTHONPATH`` (see ``smq&jgy/scripts/_env.sh``) and overrides the
``rlt_project/rlt_reproduce/src`` copy at runtime. The client code is kept identical
between the two copies (only this docstring differs) — apply any change to both.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from rlt.rl.ws_protocol import ensure_observation_images_jpeg

logger = logging.getLogger(__name__)


@dataclass
class InferResult:
    action_chunk: np.ndarray  # (H, action_dim)
    reference_action: np.ndarray  # (H, action_dim)
    state: np.ndarray | None = None  # encoded [z_rl || proprio] from GPU
    meta: dict[str, Any] | None = None


class GPUClient(ABC):
    @abstractmethod
    def infer(self, observation: dict[str, Any]) -> InferResult:
        """Send obs to GPU; receive action chunk + VLA reference."""

    @abstractmethod
    def send_transition(self, transition: dict[str, Any]) -> dict[str, Any]:
        """Send one RL transition for learner update on GPU."""

    def encode(self, observation: dict[str, Any]) -> np.ndarray:
        """Encode an observation into the RL state x = [z_rl || proprio].

        Used to build the per-step state stream (x_0..x_T) so the robot can assemble
        paper-faithful chunk transitions with an exact C-step gap between
        ``state`` and ``next_state``.
        """
        raise NotImplementedError

    def ping(self) -> dict[str, Any]:
        return {"type": "pong", "mock": True}

    def close(self) -> None:
        """Release network resources if any."""


class MockGPUClient(GPUClient):
    """Local stub: zero actions, no network. For robot-side smoke tests."""

    def __init__(self, *, action_dim: int = 7, chunk_length: int = 10, state_dim: int = 32) -> None:
        self.action_dim = action_dim
        self.chunk_length = chunk_length
        self.state_dim = state_dim
        self._transitions: list[dict] = []

    def _dummy_state(self) -> np.ndarray:
        return np.zeros(self.state_dim, dtype=np.float32)

    def infer(self, observation: dict[str, Any]) -> InferResult:
        del observation
        chunk = np.zeros((self.chunk_length, self.action_dim), dtype=np.float32)
        return InferResult(
            action_chunk=chunk,
            reference_action=chunk.copy(),
            state=self._dummy_state(),
            meta={"policy_mode": "mock"},
        )

    def send_transition(self, transition: dict[str, Any]) -> dict[str, Any]:
        self._transitions.append(transition)
        return {
            "buffer_size": len(self._transitions),
            "updated": True,
            "next_state": self._dummy_state().tolist(),
        }

    def encode(self, observation: dict[str, Any]) -> np.ndarray:
        del observation
        return self._dummy_state()

    def ping(self) -> dict[str, Any]:
        return {"type": "pong", "mock": True, "buffer_size": len(self._transitions)}


class WebsocketRLClient(GPUClient):
    """Persistent websocket client for ``rl_server.py`` on the GPU host.

    Keeps one connection open for the whole actor session (20 Hz friendly).
    Protocol: ``ONLINE_RL_TASKS.md``.
    """

    def __init__(
        self,
        host: str,
        port: int = 8765,
        *,
        timeout_sec: float = 120.0,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_sec = timeout_sec
        self.image_size = image_size
        self._uri = f"ws://{host}:{port}"
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ws = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._infer_debug_done = False

    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        def _runner() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._ready.set()
            self._loop.run_forever()

        self._ready.clear()
        self._thread = threading.Thread(target=_runner, name="gpu-ws-client", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    async def _connect(self):
        import websockets

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        # ping_interval=None: the server blocks its event loop during the multi-minute
        # VLA infer, so keepalive pings would false-timeout and drop the connection.
        # We rely on the explicit per-request recv timeout (self.timeout_sec) instead.
        self._ws = await websockets.connect(
            self._uri,
            open_timeout=self.timeout_sec,
            ping_interval=None,
            max_size=None,
        )

    async def _request_async(self, payload: dict[str, Any]) -> dict[str, Any]:
        import websockets

        if self._ws is None:
            await self._connect()
        assert self._ws is not None
        try:
            await self._ws.send(json.dumps(payload, default=_json_default))
            raw = await asyncio.wait_for(self._ws.recv(), timeout=self.timeout_sec)
            return json.loads(raw)
        except (websockets.ConnectionClosed, OSError) as exc:
            logger.warning("GPU websocket dropped (%s), reconnecting", exc)
            await self._connect()
            assert self._ws is not None
            await self._ws.send(json.dumps(payload, default=_json_default))
            raw = await asyncio.wait_for(self._ws.recv(), timeout=self.timeout_sec)
            return json.loads(raw)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._ensure_thread()
        assert self._loop is not None
        with self._lock:
            fut = asyncio.run_coroutine_threadsafe(self._request_async(payload), self._loop)
            return fut.result(timeout=self.timeout_sec + 5.0)

    def infer(self, observation: dict[str, Any]) -> InferResult:
        proprio = observation.get("proprio")
        if proprio is None:
            raise ValueError("observation must include proprio")

        prepared = ensure_observation_images_jpeg(
            observation,
            image_size=self.image_size,
        )
        if not self._infer_debug_done:
            self._infer_debug_done = True
            images = prepared.get("images") or {}
            jpegs = prepared.get("images_jpeg") or {}
            proprio_arr = np.asarray(proprio, dtype=np.float32)
            logger.info(
                "first infer obs keys=%s images keys=%s images_jpeg keys=%s "
                "jpeg lens external=%s wrist=%s proprio shape=%s",
                sorted(prepared.keys()),
                sorted(images.keys()),
                sorted(jpegs.keys()),
                len(jpegs.get("external", "")),
                len(jpegs.get("wrist", "")),
                proprio_arr.shape,
            )

        msg: dict[str, Any] = {
            "type": "infer",
            "proprio": np.asarray(proprio, dtype=np.float32),
            "language": str(prepared.get("language", "")),
        }
        if prepared.get("images_jpeg"):
            msg["images_jpeg"] = prepared["images_jpeg"]
        elif prepared.get("images"):
            raise RuntimeError(
                "infer payload missing images_jpeg despite raw images present — "
                f"keys={sorted(prepared['images'])}"
            )
        else:
            raise RuntimeError(
                "infer payload missing images_jpeg (external + wrist). "
                "Robot actor did not receive RealSense RGB frames — check camera cache / "
                "wait_for_rgb_frames in actor_loop."
            )

        resp = self._request(msg)
        if resp.get("type") == "error":
            raise RuntimeError(resp.get("message", "infer failed"))
        action = np.asarray(resp["action_chunk"], dtype=np.float32)
        ref = np.asarray(resp.get("reference_action", resp["action_chunk"]), dtype=np.float32)
        meta = {
            k: v
            for k, v in resp.items()
            if k not in ("type", "action_chunk", "reference_action", "state")
        }
        state = np.asarray(resp["state"], dtype=np.float32) if "state" in resp else None
        return InferResult(
            action_chunk=action,
            reference_action=ref,
            state=state,
            meta=meta or None,
        )

    def encode(self, observation: dict[str, Any]) -> np.ndarray:
        proprio = observation.get("proprio")
        if proprio is None:
            raise ValueError("encode observation must include proprio")
        prepared = ensure_observation_images_jpeg(observation, image_size=self.image_size)
        msg: dict[str, Any] = {
            "type": "encode",
            "proprio": np.asarray(proprio, dtype=np.float32),
            "language": str(prepared.get("language", "")),
        }
        if prepared.get("images_jpeg"):
            msg["images_jpeg"] = prepared["images_jpeg"]
        else:
            raise RuntimeError(
                "encode payload missing images_jpeg (external + wrist) — "
                "per-step state encoding needs RealSense RGB frames."
            )
        resp = self._request(msg)
        if resp.get("type") == "error":
            raise RuntimeError(resp.get("message", "encode failed"))
        return np.asarray(resp["state"], dtype=np.float32)

    def send_transition(self, transition: dict[str, Any]) -> dict[str, Any]:
        msg = {"type": "transition", **transition}
        return self._request(msg)

    def ping(self) -> dict[str, Any]:
        return self._request({"type": "ping"})

    def close(self) -> None:
        if self._loop is None or self._ws is None:
            return

        async def _close() -> None:
            if self._ws is not None:
                await self._ws.close()

        try:
            fut = asyncio.run_coroutine_threadsafe(_close(), self._loop)
            fut.result(timeout=5.0)
        except Exception:
            pass
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def create_gpu_client(
    raw: dict,
    *,
    mock: bool = False,
    host_override: str | None = None,
) -> GPUClient:
    rl = raw.get("online_rl", {})
    gpu = raw.get("gpu_server", {})
    action_dim = rl.get("action_dim", 7)
    chunk_length = rl.get("chunk_length", 10)

    env_host = host_override or os.environ.get("GPU_SERVER_HOST")
    env_mock = os.environ.get("GPU_SERVER_MOCK", "").lower()
    use_mock = mock or (env_mock in ("1", "true", "yes"))
    if not use_mock:
        use_mock = bool(gpu.get("mock", False)) and not env_host

    if use_mock:
        return MockGPUClient(action_dim=action_dim, chunk_length=chunk_length)

    host = env_host or gpu.get("host")
    if not host:
        logger.warning("gpu_server.host not set; falling back to MockGPUClient")
        return MockGPUClient(action_dim=action_dim, chunk_length=chunk_length)

    port = int(os.environ.get("GPU_SERVER_PORT", gpu.get("port", 8765)))
    timeout_sec = float(os.environ.get("GPU_INFER_TIMEOUT_SEC", gpu.get("infer_timeout_sec", 180.0)))
    cam_cfg = raw.get("cameras", {})
    img_size_raw = cam_cfg.get("image_size")
    image_size = tuple(img_size_raw) if img_size_raw else None
    logger.info("Connecting to GPU RL server ws://%s:%d (timeout=%.0fs)", host, port, timeout_sec)
    return WebsocketRLClient(host=str(host), port=port, timeout_sec=timeout_sec, image_size=image_size)
