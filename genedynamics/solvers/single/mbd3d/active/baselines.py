"""
Baseline view scorers (Random, MaxDistance) for active view selection.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..types import SceneParams
from .view_scorer import ViewScorer


class RandomScorer(ViewScorer):
    """Uniformly random scores; ties broken by shuffle."""

    name = "random"

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def score(
        self,
        posterior_scenes: List[SceneParams],
        candidate_poses: np.ndarray,
        intrinsics: np.ndarray,
        renderer,
        *,
        selected_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        scores = self._rng.standard_normal(candidate_poses.shape[0]).astype(np.float32)
        if selected_mask is not None:
            scores[selected_mask.astype(bool)] = -np.inf
        return scores


class MaxDistanceScorer(ViewScorer):
    """
    Geometric heuristic: pick the candidate farthest (in position) from the
    already-selected views. Good baseline for baseline-diversity.

    If no views are selected yet, falls back to uniform random.
    """

    name = "max_distance"

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def score(
        self,
        posterior_scenes: List[SceneParams],
        candidate_poses: np.ndarray,
        intrinsics: np.ndarray,
        renderer,
        *,
        selected_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        C = int(candidate_poses.shape[0])
        pos = np.asarray(candidate_poses[:, :3], dtype=np.float32)

        if selected_mask is None or not np.any(selected_mask):
            return self._rng.standard_normal(C).astype(np.float32)

        selected = pos[selected_mask.astype(bool)]
        # For each candidate, min-distance to any selected view.
        diffs = pos[:, None, :] - selected[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)        # (C, S)
        min_dist = dists.min(axis=-1)                 # (C,)
        scores = min_dist.copy()
        scores[selected_mask.astype(bool)] = -np.inf
        return scores.astype(np.float32)
