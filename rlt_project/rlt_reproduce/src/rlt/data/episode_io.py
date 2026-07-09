"""Save/load critical-phase episodes as NPZ bundles."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rlt.data.schema import CriticalPhaseEpisode, EpisodeMetadata


def _episode_index(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("ep_"):
        return None
    try:
        return int(stem[3:])
    except ValueError:
        return None


def count_episodes(episodes_dir: str | Path) -> int:
    return len(list_episodes(episodes_dir))


def next_episode_index(episodes_dir: str | Path) -> int:
    """Next unused ep_XXXXX index (max existing + 1, or 0 if empty)."""
    episodes_dir = Path(episodes_dir)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    indices = [_episode_index(p) for p in episodes_dir.glob("*.npz")]
    indices = [i for i in indices if i is not None]
    return (max(indices) + 1) if indices else 0


def save_episode(episode: CriticalPhaseEpisode, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_id = f"ep_{next_episode_index(out_dir):05d}"
    path = out_dir / f"{episode_id}.npz"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing episode: {path}")
    meta = asdict(episode.metadata)
    np.savez_compressed(
        path,
        timestamps=episode.timestamps,
        proprio=episode.proprio,
        actions=episode.actions,
        rewards=episode.rewards,
        dones=episode.dones.astype(np.uint8),
        is_human=episode.is_human.astype(np.uint8),
        reference_actions=episode.reference_actions
        if episode.reference_actions is not None
        else np.array([]),
        images_wrist=episode.images_wrist if episode.images_wrist is not None else np.array([]),
        images_external=episode.images_external if episode.images_external is not None else np.array([]),
        metadata_json=json.dumps(meta),
    )
    return path


def load_episode(path: str | Path) -> CriticalPhaseEpisode:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        meta = EpisodeMetadata(**json.loads(str(data["metadata_json"])))
        ref = data["reference_actions"]
        wrist = data["images_wrist"]
        ext = data["images_external"]
        return CriticalPhaseEpisode(
            metadata=meta,
            timestamps=data["timestamps"],
            proprio=data["proprio"],
            actions=data["actions"],
            rewards=data["rewards"],
            dones=data["dones"].astype(bool),
            is_human=data["is_human"].astype(bool),
            reference_actions=ref if ref.size else None,
            images_wrist=wrist if wrist.size else None,
            images_external=ext if ext.size else None,
        )


def list_episodes(episodes_dir: str | Path) -> list[Path]:
    return sorted(Path(episodes_dir).glob("*.npz"))
