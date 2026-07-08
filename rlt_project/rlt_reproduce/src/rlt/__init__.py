"""RLT (RL Token) community reproduction for Physical Intelligence pi0.6 VLAs."""

from pathlib import Path

__version__ = "0.1.0"

# GPU-synced modules live under ``smq&jgy/src/rlt/``; prepend so they override copies here.
_smq_root = Path(__file__).resolve().parents[4]
_overlay = _smq_root / "src" / "rlt"
if _overlay.is_dir():
    __path__.insert(0, str(_overlay))
