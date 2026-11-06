"""Torch trajectory filter driven by a task-owned feasibility evaluator.

The filter is deliberately task agnostic.  It contracts an action trajectory
towards the normalized zero anchor and asks the wrapped dynamics binding for
feasibility evidence.  Task geometry, costs and constraint semantics remain in
the binding; MDOC retains ownership of where the filter is applied.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import Any

import torch

from .base import ConstraintFilter


class TorchTrajectoryFeasibilityFilter(ConstraintFilter):
    """Contract infeasible trajectory proposals towards a trusted anchor."""

    requires_model_call_counter = True

    def __init__(self, contraction_factors: Iterable[float] = (0.5, 0.25, 0.0)):
        factors = tuple(float(value) for value in contraction_factors)
        if (
            not factors
            or any(not 0.0 <= value < 1.0 for value in factors)
            or any(
                left <= right for left, right in itertools.pairwise(factors)
            )
        ):
            raise ValueError(
                "Trajectory contraction factors must be strictly decreasing "
                "values in [0, 1)"
            )
        self.contraction_factors = factors

    @staticmethod
    def _evaluate(env: Any, actions: torch.Tensor):
        if not hasattr(env, "evaluate_action_sequences"):
            raise TypeError(
                "Trajectory feasibility filtering requires "
                "evaluate_action_sequences"
            )
        result = env.evaluate_action_sequences(actions)
        batch = actions.shape[0]
        maximum_violation = getattr(result, "maximum_violation", None)
        predicted_feasible = getattr(result, "predicted_feasible", None)
        if (
            not isinstance(maximum_violation, torch.Tensor)
            or maximum_violation.shape != (batch,)
            or not torch.isfinite(maximum_violation).all()
            or not isinstance(predicted_feasible, torch.Tensor)
            or predicted_feasible.shape != (batch,)
            or predicted_feasible.dtype is not torch.bool
        ):
            raise ValueError(
                "Trajectory evaluator returned invalid feasibility evidence"
            )
        return maximum_violation, predicted_feasible

    def apply_actions(
        self,
        x0: Any,
        actions: Any,
        *,
        env: Any,
        obstacles: Any = None,
        schedule_state: Any = None,
        schedule_params: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del x0, obstacles, schedule_state, schedule_params, kwargs
        if not isinstance(actions, torch.Tensor) or actions.ndim not in {2, 3}:
            raise TypeError(
                "Torch trajectory filtering requires [horizon, action] or "
                "[batch, horizon, action] tensors"
            )
        squeeze = actions.ndim == 2
        batch = actions.unsqueeze(0) if squeeze else actions
        if not torch.isfinite(batch).all():
            raise ValueError("Trajectory filter actions must be finite")

        best = batch.clone()
        best_violation, feasible = self._evaluate(env, batch)
        unresolved = ~feasible
        for factor in self.contraction_factors:
            indices = torch.nonzero(unresolved, as_tuple=False).flatten()
            if not len(indices):
                break
            trial = batch.index_select(0, indices) * factor
            trial_violation, trial_feasible = self._evaluate(env, trial)
            previous = best_violation.index_select(0, indices)
            improved = trial_violation < previous
            if bool(improved.any()):
                improved_indices = indices[improved]
                best[improved_indices] = trial[improved]
                best_violation[improved_indices] = trial_violation[improved]
            if bool(trial_feasible.any()):
                feasible_indices = indices[trial_feasible]
                best[feasible_indices] = trial[trial_feasible]
                best_violation[feasible_indices] = trial_violation[trial_feasible]
                unresolved[feasible_indices] = False
        return best[0] if squeeze else best


__all__ = ["TorchTrajectoryFeasibilityFilter"]
