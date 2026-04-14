"""
View scorers for active view selection.

A `ViewScorer.score(...)` returns a (C,) array of scores for C candidate views;
higher is better. The selection loop calls top-k on those scores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from ..types import SceneParams


class ViewScorer(ABC):
    """Abstract base for active view scorers."""

    name: str = "base"

    @abstractmethod
    def score(
        self,
        posterior_scenes: List[SceneParams],
        candidate_poses: np.ndarray,            # (C, 7)
        intrinsics: np.ndarray,                 # (3, 3) or (C, 3, 3)
        renderer,
        *,
        selected_mask: Optional[np.ndarray] = None,  # (C,) bool; already-picked views
    ) -> np.ndarray:
        """Return a (C,) float array; higher = more informative."""
        raise NotImplementedError


class PosteriorVarianceScorer(ViewScorer):
    """
    Information-gain proxy:  score(v) = mean_pixel( Var_m( Π(θ^m; v) ) ).

    When posterior samples disagree on what view v looks like, that view is
    highly informative. This is an ensemble-based proxy for mutual information
    (cf. BALD / predictive uncertainty acquisition).
    """

    name = "posterior_variance"

    def __init__(self, *, color_channel_reduce: str = "mean", batch_size: int = 4):
        self.color_channel_reduce = color_channel_reduce  # "mean" | "sum"
        self.batch_size = int(batch_size)

    def score(
        self,
        posterior_scenes: List[SceneParams],
        candidate_poses: np.ndarray,
        intrinsics: np.ndarray,
        renderer,
        *,
        selected_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not posterior_scenes:
            raise ValueError("PosteriorVarianceScorer requires ≥1 posterior sample.")

        C = int(candidate_poses.shape[0])
        renders_per_scene: List[np.ndarray] = []
        # (M, C, H, W, 3). Render per scene to avoid blowing up GPU memory.
        for sp in posterior_scenes:
            imgs = np.asarray(
                renderer.render(sp, candidate_poses, intrinsics=intrinsics),
                dtype=np.float32,
            )
            renders_per_scene.append(np.clip(imgs, 0.0, 1.0))
        stack = np.stack(renders_per_scene, axis=0)  # (M, C, H, W, 3)

        # Variance across M, then reduce over H, W, channels.
        var = np.var(stack, axis=0)                  # (C, H, W, 3)
        if self.color_channel_reduce == "sum":
            var = var.sum(axis=-1)
        else:
            var = var.mean(axis=-1)
        scores = var.mean(axis=(-2, -1))             # (C,)

        if selected_mask is not None:
            scores = scores.copy()
            scores[selected_mask.astype(bool)] = -np.inf
        return scores.astype(np.float32)
