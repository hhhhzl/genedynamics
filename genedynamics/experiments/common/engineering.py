"""
Engineering enhancements: checkpointing, logging, reproducibility.

Task-agnostic utilities for experiment robustness.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _to_serializable(obj: Any) -> Any:
    """Convert to JSON-serializable form."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    return obj


@dataclass
class CheckpointState:
    """State for checkpointing."""

    step: int
    seed: int
    config_hash: str
    wall_time: float
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class CheckpointManager:
    """
    Checkpoint manager for experiment resume.

    Supports step-based (pruned) and keyed checkpoints (for resume).
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        *,
        max_keep: int = 3,
        prefix: str = "ckpt",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.prefix = prefix

    def save(self, state: CheckpointState, *, key: Optional[str] = None) -> Path:
        """Save checkpoint. key= for keyed (no prune), used for resume."""
        if key is not None:
            path = self.checkpoint_dir / f"{self.prefix}_{key}.json"
        else:
            path = self.checkpoint_dir / f"{self.prefix}_{state.step:06d}.json"
        with open(path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
        if key is None:
            self._prune()
        return path

    def load_latest(self, *, key: Optional[str] = None) -> Optional[CheckpointState]:
        """Load checkpoint. key= for keyed (resume)."""
        if key is not None:
            path = self.checkpoint_dir / f"{self.prefix}_{key}.json"
            if not path.exists():
                return None
            with open(path) as f:
                return CheckpointState.from_dict(json.load(f))
        pattern = f"{self.prefix}_[0-9]*.json"
        files = [p for p in self.checkpoint_dir.glob(pattern) if p.stem.split("_")[-1].isdigit()]
        if not files:
            return None
        files = sorted(files, key=lambda p: int(p.stem.split("_")[-1]))
        with open(files[-1]) as f:
            return CheckpointState.from_dict(json.load(f))

    def _prune(self) -> None:
        pattern = f"{self.prefix}_[0-9]*.json"
        files = [p for p in self.checkpoint_dir.glob(pattern) if p.stem.split("_")[-1].isdigit()]
        if not files:
            return
        files = sorted(files, key=lambda p: int(p.stem.split("_")[-1]), reverse=True)
        for f in files[self.max_keep :]:
            try:
                f.unlink()
            except OSError:
                pass


class ExperimentLogger:
    """Structured JSON-lines logger for experiments."""

    def __init__(self, log_path: Path, *, buffer_size: int = 1):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        self._buffer: List[Dict[str, Any]] = []

    def log(self, event: str, **kwargs: Any) -> None:
        entry = {"timestamp": time.time(), "event": event, **kwargs}
        self._buffer.append(_to_serializable(entry))
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        with open(self.log_path, "a") as f:
            for entry in self._buffer:
                f.write(json.dumps(entry) + "\n")
        self._buffer.clear()


def config_hash_dict(config: Dict[str, Any]) -> str:
    """Stable hash for config (reproducibility)."""
    data = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def set_seed(seed: int) -> None:
    """Set global seeds for reproducibility."""
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
