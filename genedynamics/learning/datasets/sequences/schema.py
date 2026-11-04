"""A transition-aligned episode keeps both pre- and post-action states."""

from dataclasses import dataclass
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class Episode:
    identifier: str
    source_sequence: str
    split: str
    arrays: Mapping[str, np.ndarray]
    metadata: Mapping

    def __post_init__(self):
        required = {"state", "next_state", "action", "terminated", "truncated"}
        if not required <= self.arrays.keys():
            raise ValueError(f"Missing episode fields: {sorted(required - self.arrays.keys())}")
        lengths = {len(value) for value in self.arrays.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
            raise ValueError("All episode fields must share a positive transition axis")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("Unknown dataset split")
        if not self.identifier or not self.source_sequence:
            raise ValueError("Episode and original source-sequence identifiers are required")
        for name, value in self.arrays.items():
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise ValueError(f"Object arrays are forbidden in {name}")
            if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
                raise ValueError(f"Non-finite data in {name}; represent invalid transitions with explicit masks")
        if self.arrays["state"].shape != self.arrays["next_state"].shape:
            raise ValueError("state and next_state must have identical shapes")
        done = np.asarray(self.arrays["terminated"], bool) | np.asarray(self.arrays["truncated"], bool)
        if done.ndim != 1 or done[:-1].any():
            raise ValueError("An episode cannot contain transitions after an episode end")

    def __len__(self):
        return len(self.arrays["action"])


@dataclass(frozen=True)
class SequenceBatch:
    arrays: Mapping[str, np.ndarray]
    valid: np.ndarray
    loss_mask: np.ndarray
    episode_identifiers: tuple[str, ...]
    start_indices: np.ndarray
