"""
Active view selection for MBD3D (exp2B).

Given a posterior over scene params {θ^(1), ..., θ^(M)} (from multiple MBD chains
or posterior samples) and a pool of candidate camera poses, pick the next view(s)
that maximize expected information gain.

Modules
-------
- `view_scorer.ViewScorer`: abstract interface; returns a score per candidate view.
- `view_scorer.PosteriorVarianceScorer`: information-theoretic baseline that
  uses Var_m(Π(θ^m; v)) as a proxy for mutual information.
- `baselines.RandomScorer`, `baselines.MaxDistanceScorer`: geometric baselines.
- `selection_loop.ActiveSelectionLoop`: orchestrates iterate(select → add → re-run).
"""

from .view_scorer import ViewScorer, PosteriorVarianceScorer
from .baselines import RandomScorer, MaxDistanceScorer
from .selection_loop import ActiveSelectionLoop, SelectionRecord

__all__ = [
    "ViewScorer",
    "PosteriorVarianceScorer",
    "RandomScorer",
    "MaxDistanceScorer",
    "ActiveSelectionLoop",
    "SelectionRecord",
]
