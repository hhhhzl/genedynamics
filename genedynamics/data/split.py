"""
Split utilities for NeRF Synthetic and 3DGS experiments.

Supports fixed train/val/test splits and reproducible view sampling.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def sample_view_indices(
    n_frames: int,
    stride: int = 1,
    max_views: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample frame indices for train/val/test.

    Args:
        n_frames: Total number of frames
        stride: Step between consecutive indices (1 = all)
        max_views: Cap number of views (None = no cap)
        shuffle_seed: If set, shuffle then take first max_views (reproducible)

    Returns:
        Sorted indices (int32)
    """
    stride = max(1, int(stride))
    idx = np.arange(0, n_frames, stride, dtype=np.int32)

    if shuffle_seed is not None:
        rng = np.random.default_rng(int(shuffle_seed))
        rng.shuffle(idx)

    if max_views is not None and max_views > 0 and idx.size > max_views:
        idx = idx[: int(max_views)]

    return np.sort(idx)


def compute_split_ratios(
    n_total: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> tuple[int, int, int]:
    """
    Compute train/val/test counts from ratios.

    Ensures at least 1 frame per split when possible.
    """
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(0, int(n_total * val_ratio))
    n_test = max(0, n_total - n_train - n_val)
    if n_test < 0:
        n_test = 0
        n_val = n_total - n_train
    return n_train, n_val, n_test


def split_indices(
    indices: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split indices into train/val/test.

    Uses first n_train, next n_val, last n_test (no shuffle by default).
    If seed is set, shuffles indices first.
    """
    idx = np.asarray(indices, dtype=np.int32).ravel()
    if seed is not None:
        rng = np.random.default_rng(int(seed))
        rng.shuffle(idx)

    i = 0
    train_idx = idx[i : i + n_train]
    i += n_train
    val_idx = idx[i : i + n_val]
    i += n_val
    test_idx = idx[i : i + n_test]

    return train_idx, val_idx, test_idx
