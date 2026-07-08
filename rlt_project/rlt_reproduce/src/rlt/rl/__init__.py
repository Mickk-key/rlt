"""Online RL modules; GPU-synced client under ``smq&jgy/src/rlt/rl/``."""

from pathlib import Path

_overlay = Path(__file__).resolve().parents[5] / "src" / "rlt" / "rl"
if _overlay.is_dir():
    __path__.insert(0, str(_overlay))
