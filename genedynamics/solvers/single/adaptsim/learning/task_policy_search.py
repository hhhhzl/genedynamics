"""Task-policy optimization inside an AdaptSim simulator distribution.

The author's policy optimizer is task-specific (one-step NAF for pushing).
This bounded CEM optimizer is an explicit long-horizon task adaptation, not
an upstream policy-training implementation.  It charges every simulation
rollout through the task-provided objective callback.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..adaptsim import ParameterDistribution
from .adaptation import AdaptationTraining


@dataclass(frozen=True)
class PolicySearchEvaluation:
    reward: float
    simulation_steps: int
    episodes: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not np.isfinite(self.reward) or self.simulation_steps < 0:
            raise ValueError("policy-search reward and simulation steps must be valid")


class CEMTaskPolicySearch:
    """Search a bounded task-policy vector under sampled simulator parameters."""

    def __init__(
        self,
        lower: Sequence[float],
        upper: Sequence[float],
        anchor: Sequence[float],
        sample_tasks: Callable[[ParameterDistribution, np.random.Generator, int], Any],
        evaluate_policy: Callable[[np.ndarray, Any, int], PolicySearchEvaluation],
        *,
        population: int,
        elite: int,
        iterations: int,
        tasks_per_candidate: int,
        coordinate_probes: Sequence[int] = (),
        probe_fraction: float = 0.1,
    ) -> None:
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.anchor = np.asarray(anchor, dtype=np.float64)
        if (self.lower.ndim != 1 or self.lower.shape != self.upper.shape
                or self.anchor.shape != self.lower.shape
                or not np.isfinite(self.lower).all()
                or not np.isfinite(self.upper).all()
                or not np.isfinite(self.anchor).all()
                or np.any(self.lower >= self.upper)
                or np.any(self.anchor < self.lower)
                or np.any(self.anchor > self.upper)):
            raise ValueError("finite ordered bounds and in-range anchor required")
        if not 2 <= population or not 1 <= elite < population:
            raise ValueError("CEM requires population >= 2 and 1 <= elite < population")
        if iterations <= 0 or tasks_per_candidate <= 0:
            raise ValueError("CEM iterations and task count must be positive")
        self.coordinate_probes = tuple(int(index) for index in coordinate_probes)
        if (len(set(self.coordinate_probes)) != len(self.coordinate_probes)
                or any(index < 0 or index >= len(self.anchor) for index in self.coordinate_probes)
                or population < 1 + 2 * len(self.coordinate_probes)
                or not np.isfinite(probe_fraction) or probe_fraction <= 0 or probe_fraction >= 0.5):
            raise ValueError("coordinate probes require valid indices, fraction, and population")
        self.probe_fraction = float(probe_fraction)
        self.sample_tasks = sample_tasks
        self.evaluate_policy = evaluate_policy
        self.population = int(population)
        self.elite = int(elite)
        self.iterations = int(iterations)
        self.tasks_per_candidate = int(tasks_per_candidate)
        self.last_ledger: tuple[dict[str, Any], ...] = ()

    def __call__(
        self, distribution: ParameterDistribution, warm_start: np.ndarray | None,
        seed: int,
    ) -> AdaptationTraining:
        rng = np.random.default_rng(seed)
        mean = self.anchor.copy() if warm_start is None else np.asarray(warm_start, dtype=np.float64).copy()
        if mean.shape != self.anchor.shape or not np.isfinite(mean).all():
            raise ValueError("warm-start policy vector shape or values differ")
        mean = np.clip(mean, self.lower, self.upper)
        std = (self.upper - self.lower) / 4
        best_reward = -np.inf
        best = mean.copy()
        consumed = 0
        ledger: list[dict[str, Any]] = []
        for iteration in range(self.iterations):
            # Every candidate sees the same sampled simulator tasks in one
            # CEM iteration; this removes task-sampling noise from ranking.
            tasks = self.sample_tasks(distribution, rng, self.tasks_per_candidate)
            candidates = np.clip(
                rng.normal(mean, std, size=(self.population, len(mean))),
                self.lower, self.upper,
            )
            candidates[0] = mean
            for probe_index, coordinate in enumerate(self.coordinate_probes):
                delta = self.probe_fraction * (self.upper[coordinate] - self.lower[coordinate])
                for offset, sign in enumerate((-1, 1)):
                    candidate = mean.copy()
                    candidate[coordinate] = np.clip(
                        candidate[coordinate] + sign * delta,
                        self.lower[coordinate], self.upper[coordinate],
                    )
                    candidates[1 + 2 * probe_index + offset] = candidate
            if (iteration == 0 and self.population > 2
                    and not np.array_equal(mean, self.anchor)
                    and 1 + 2 * len(self.coordinate_probes) < self.population):
                candidates[1 + 2 * len(self.coordinate_probes)] = self.anchor
            rewards = np.empty(self.population, dtype=np.float64)
            for index, candidate in enumerate(candidates):
                evaluation = self.evaluate_policy(
                    candidate.copy(), tasks, seed + iteration * self.population + index
                )
                if not isinstance(evaluation, PolicySearchEvaluation):
                    raise TypeError("evaluate_policy must return PolicySearchEvaluation")
                consumed += evaluation.simulation_steps
                rewards[index] = evaluation.reward
                ledger.append({
                    "iteration": iteration,
                    "candidate": candidate.tolist(),
                    "reward": float(evaluation.reward),
                    "simulation_steps": evaluation.simulation_steps,
                    "episodes": [dict(row) for row in evaluation.episodes],
                })
                if evaluation.reward > best_reward:
                    best_reward = float(evaluation.reward)
                    best = candidate.copy()
            ranked = np.argsort(-rewards, kind="stable")[:self.elite]
            selected = candidates[ranked]
            mean = selected.mean(axis=0)
            std = np.maximum(selected.std(axis=0), (self.upper - self.lower) / 100)
        self.last_ledger = tuple(ledger)
        return AdaptationTraining(policy=best, simulation_steps=consumed)


__all__ = ["CEMTaskPolicySearch", "PolicySearchEvaluation"]
