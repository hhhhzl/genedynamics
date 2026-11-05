"""Bounded black-box search for task-adapted ASID stages.

The public release contains the Fisher objective but no reusable optimizer
for our tasks. This CEM implementation is independent code; its callback owns
all simulator interactions and must return a complete cost ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class ASIDSearchResult:
    best: np.ndarray
    best_score: float
    mean: np.ndarray
    std: np.ndarray
    ledger: tuple[dict[str, Any], ...]


class ASIDBoundedCEM:
    """Optimize a finite vector while retaining every candidate evaluation."""

    def __init__(self, lower: Sequence[float], upper: Sequence[float],
                 anchor: Sequence[float], *, population: int, elite: int,
                 iterations: int, maximize: bool) -> None:
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.anchor = np.asarray(anchor, dtype=np.float64)
        if (self.lower.ndim != 1 or self.lower.size == 0
                or self.lower.shape != self.upper.shape
                or self.anchor.shape != self.lower.shape
                or not np.isfinite(self.lower).all()
                or not np.isfinite(self.upper).all()
                or not np.isfinite(self.anchor).all()
                or np.any(self.lower >= self.upper)
                or np.any(self.anchor < self.lower)
                or np.any(self.anchor > self.upper)):
            raise ValueError("finite bounded search vector and in-range anchor required")
        if (isinstance(population, bool) or isinstance(elite, bool)
                or isinstance(iterations, bool) or population < 2
                or not 1 <= elite < population or iterations < 1):
            raise ValueError("invalid CEM population, elite count, or iterations")
        self.population = population
        self.elite = elite
        self.iterations = iterations
        self.maximize = bool(maximize)

    def run(self, evaluate: Callable[[np.ndarray, int, int],
                             tuple[float, dict[str, Any]]], *, seed: int
            ) -> ASIDSearchResult:
        rng = np.random.default_rng(seed)
        mean = self.anchor.copy()
        std = (self.upper - self.lower) / 4.0
        best = mean.copy()
        best_score = -np.inf if self.maximize else np.inf
        ledger: list[dict[str, Any]] = []
        for iteration in range(self.iterations):
            candidates = np.clip(
                rng.normal(mean, std, (self.population, len(mean))),
                self.lower, self.upper,
            )
            candidates[0] = mean
            scores = np.empty(self.population, dtype=np.float64)
            for index, candidate in enumerate(candidates):
                score, details = evaluate(candidate.copy(), iteration, index)
                if (not np.isfinite(score) or not isinstance(details, dict)
                        or "episodes" not in details):
                    raise ValueError("each candidate needs finite score and episode ledger")
                scores[index] = float(score)
                ledger.append({"iteration": iteration, "candidate_index": index,
                               "candidate": candidate.tolist(),
                               "score": float(score), **details})
                if ((score > best_score) if self.maximize else (score < best_score)):
                    best_score = float(score)
                    best = candidate.copy()
            ranking = np.argsort(-scores if self.maximize else scores, kind="stable")
            selected = candidates[ranking[:self.elite]]
            mean = selected.mean(axis=0)
            std = np.maximum(selected.std(axis=0), (self.upper - self.lower) / 100.0)
        return ASIDSearchResult(best, best_score, mean, std, tuple(ledger))


__all__ = ["ASIDBoundedCEM", "ASIDSearchResult"]
