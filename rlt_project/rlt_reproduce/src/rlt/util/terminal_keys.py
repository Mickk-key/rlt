"""Non-blocking terminal key reads (SSH-safe, no DISPLAY required)."""

from __future__ import annotations

import select
import sys
import termios
import tty
from contextlib import contextmanager


def stdin_is_tty() -> bool:
    return sys.stdin.isatty()


@contextmanager
def terminal_keys():
    """Put stdin in cbreak mode for single-key polling. No-op if stdin is not a TTY."""
    if not stdin_is_tty():
        yield
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def poll_key() -> int:
    """Return ord of next key, or 0 if none pending / not a TTY."""
    if not stdin_is_tty():
        return 0
    if not select.select([sys.stdin], [], [], 0)[0]:
        return 0
    ch = sys.stdin.read(1)
    return ord(ch) if ch else 0


def flush_input() -> None:
    """Discard any keystrokes buffered on stdin (no-op if not a TTY).

    stdin is held in cbreak mode for the whole rollout, so keys typed during the
    long non-interactive windows (external reset subprocess, camera restart,
    post-reset warmup, first 30-90s CUDA-warmup infer) stay buffered and would
    otherwise be consumed at step 1 of the next episode — e.g. a stray ``q`` ends
    the episode immediately (reason=quit, steps=1). Draining before execution
    starts makes the s/f/q keys reflect only what the operator presses *during*
    the episode."""
    if not stdin_is_tty():
        return
    try:
        while select.select([sys.stdin], [], [], 0)[0]:
            if not sys.stdin.read(1):
                break
    except (OSError, ValueError):
        pass
