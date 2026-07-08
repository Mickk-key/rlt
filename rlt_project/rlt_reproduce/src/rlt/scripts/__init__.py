"""RLT CLI scripts; GPU-synced entries live under ``smq&jgy/src/rlt/scripts/``."""

from pathlib import Path

_overlay = Path(__file__).resolve().parents[5] / "src" / "rlt" / "scripts"
if _overlay.is_dir():
    __path__.insert(0, str(_overlay))
