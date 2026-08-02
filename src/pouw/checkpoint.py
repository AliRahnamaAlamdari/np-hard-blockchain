"""Checkpointing and resume for long simulation runs.

The 500-block fairness run and the N=12 problem-size sweep both take long enough that
losing progress matters.  A checkpoint stores the completed block records and the
config hash; resume refuses to continue from a checkpoint written under a different
configuration rather than silently mixing two parameter sets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .crypto.hashing import canonical_json, sha256_hex
from .logging_utils import get_logger

LOGGER = get_logger("pouw.checkpoint")


def config_fingerprint(config: Any) -> str:
    """Stable hash of a resolved config, used to guard resume."""
    payload = config.to_dict() if hasattr(config, "to_dict") else config
    return sha256_hex(canonical_json(payload))


@dataclass
class Checkpoint:
    """Serialisable snapshot of an in-progress run."""

    experiment: str
    config_hash: str
    seed: int
    completed_blocks: int
    records: list[dict[str, Any]]
    state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "completed_blocks": self.completed_blocks,
            "records": self.records,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Checkpoint":
        return cls(**payload)


def checkpoint_path(output_dir: Path, experiment: str, seed: int) -> Path:
    return Path(output_dir) / f"{experiment}_seed{seed}.checkpoint.json"


def save(path: Path, checkpoint: Checkpoint) -> Path:
    """Write atomically so an interrupted save cannot corrupt an earlier checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(checkpoint.to_dict(), indent=2))
    temporary.replace(path)
    LOGGER.debug("checkpoint saved: %s (%d blocks)", path, checkpoint.completed_blocks)
    return path


def load(path: Path, expected_config_hash: str | None = None) -> Checkpoint | None:
    """Load a checkpoint, or return None when there is nothing valid to resume."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        checkpoint = Checkpoint.from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, TypeError, KeyError) as error:
        LOGGER.warning("ignoring unreadable checkpoint %s: %s", path, error)
        return None
    if expected_config_hash and checkpoint.config_hash != expected_config_hash:
        LOGGER.warning(
            "ignoring checkpoint %s: written under config %s, current config is %s",
            path,
            checkpoint.config_hash[:12],
            expected_config_hash[:12],
        )
        return None
    LOGGER.info("resuming from %s at block %d", path, checkpoint.completed_blocks)
    return checkpoint


def clear(path: Path) -> None:
    path = Path(path)
    if path.exists():
        path.unlink()
