"""Human-in-the-loop sparse rewards for online RL on the robot PC."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from rlt.util.terminal_keys import poll_key


@dataclass(frozen=True)
class EpisodeOutcome:
    """Terminal signal for one RL episode."""

    reward: float
    done: bool
    reason: str  # success_key | fail_key | timeout | quit

    def to_dict(self) -> dict:
        return asdict(self)


class RewardLogger:
    """Poll keyboard for success/fail; optional step timeout as automatic fail.

    Controls (terminal must be focused):
      s = success → reward=1, done=True
      f = fail    → reward=0, done=True
      q = quit entire run (fail)
    """

    SUCCESS_KEY = ord("s")
    FAIL_KEY = ord("f")
    QUIT_KEY = ord("q")

    def __init__(
        self,
        *,
        max_steps: int = 200,
        log_dir: Path | str | None = None,
        poll_keys: bool = True,
    ) -> None:
        self.max_steps = max_steps
        self.log_dir = Path(log_dir) if log_dir else None
        self.poll_keys = poll_keys
        self._episode_index = 0

    def poll(self, step: int) -> EpisodeOutcome | None:
        """Non-blocking check: keypress or timeout. None = continue episode."""
        if self.poll_keys:
            key = poll_key()
            if key == self.SUCCESS_KEY:
                return EpisodeOutcome(reward=1.0, done=True, reason="success_key")
            if key in (self.FAIL_KEY, self.QUIT_KEY):
                reason = "quit" if key == self.QUIT_KEY else "fail_key"
                return EpisodeOutcome(reward=0.0, done=True, reason=reason)
        if step >= self.max_steps:
            return EpisodeOutcome(reward=0.0, done=True, reason="timeout")
        return None

    def log_episode(
        self,
        outcome: EpisodeOutcome,
        *,
        episode_id: str | None = None,
        steps: int = 0,
        total_reward: float = 0.0,
        extra: dict | None = None,
    ) -> Path | None:
        """Write ``logs/online_rl/rewards/ep_XXX.json``; return path if saved."""
        if self.log_dir is None:
            return None

        ep_id = episode_id or f"ep_{self._episode_index:04d}"
        self._episode_index += 1
        payload = {
            "episode_id": ep_id,
            "timestamp": time.time(),
            "steps": steps,
            "total_reward": total_reward,
            **outcome.to_dict(),
        }
        if extra:
            payload["extra"] = extra

        out_dir = self.log_dir / "rewards"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{ep_id}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
