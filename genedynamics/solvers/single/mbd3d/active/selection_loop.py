"""
Active selection loop: iterate select → add → re-run MBD.

Public surface
--------------
`ActiveSelectionLoop.run()` orchestrates K rounds:

    for round in range(K):
        1. Rank candidate views with `ViewScorer.score(...)`.
        2. Add top-n candidates to the active pool.
        3. Call `reconstruct_fn(ObservationBundle-for-active-pool)`
           → returns (posterior_scenes, eval_metrics).
        4. Record selection + metric trajectory.

The `reconstruct_fn` is plugged in by callers (so the same loop works for
MBD3D, MAP gsplat, or any other scene estimator).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from genedynamics.data import ObservationBundle

from ..types import SceneParams
from .view_scorer import ViewScorer


ReconstructFn = Callable[
    [ObservationBundle, Dict[str, Any]],  # (active_bundle, context)
    Tuple[List[SceneParams], Dict[str, float]],  # (posterior_scenes, metrics)
]


@dataclass
class SelectionRecord:
    """Per-round record emitted by ActiveSelectionLoop."""

    round_idx: int
    selected_this_round: List[int]
    active_indices: List[int]
    metrics: Dict[str, float]
    scores: Optional[np.ndarray] = None  # (C,) per-candidate, for debugging


@dataclass
class ActiveSelectionLoop:
    """Run `n_rounds` of active view selection."""

    scorer: ViewScorer
    reconstruct_fn: ReconstructFn

    # Candidate pool
    candidate_images: np.ndarray                    # (C, H, W, 3)
    candidate_poses: np.ndarray                     # (C, 7)
    intrinsics: np.ndarray

    # Loop controls
    init_indices: List[int] = field(default_factory=list)
    n_rounds: int = 5
    n_select_per_round: int = 1

    renderer: Optional[Any] = None  # needed by view_scorer

    def _gather_bundle(self, active: List[int]) -> ObservationBundle:
        imgs = self.candidate_images[active]
        poses = self.candidate_poses[active]
        return ObservationBundle(
            images=imgs, camera_poses=poses, intrinsics=self.intrinsics,
        )

    def run(self) -> List[SelectionRecord]:
        records: List[SelectionRecord] = []
        C = int(self.candidate_images.shape[0])
        selected_mask = np.zeros(C, dtype=bool)
        active: List[int] = list(self.init_indices)
        selected_mask[active] = True

        # ---- Round 0: initial reconstruction on seed views ----
        bundle = self._gather_bundle(active)
        posterior, metrics = self.reconstruct_fn(bundle, {"round": 0})
        records.append(SelectionRecord(
            round_idx=0, selected_this_round=list(self.init_indices),
            active_indices=list(active), metrics=metrics,
        ))

        for r in range(1, self.n_rounds + 1):
            scores = self.scorer.score(
                posterior, self.candidate_poses, self.intrinsics, self.renderer,
                selected_mask=selected_mask,
            )
            k = int(self.n_select_per_round)
            remaining = np.where(~selected_mask)[0]
            if len(remaining) == 0:
                break
            top_k = remaining[np.argsort(-scores[remaining])[:k]]
            for idx in top_k:
                selected_mask[idx] = True
                active.append(int(idx))

            bundle = self._gather_bundle(active)
            posterior, metrics = self.reconstruct_fn(bundle, {"round": r})
            records.append(SelectionRecord(
                round_idx=r,
                selected_this_round=top_k.tolist(),
                active_indices=list(active),
                metrics=metrics,
                scores=scores,
            ))

        return records
