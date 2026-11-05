"""Matched callback baselines reject invalid predictions on each numerical runtime."""

import numpy as np
import pytest
from genedynamics.core.backends.differentiation import make_runtime
from genedynamics.solvers.single.budgeted_repair import ControlProblem, RepairBudget
from genedynamics.solvers.single.model_mppi import ModelMPPISolver
from genedynamics.solvers.single.reference_mga import ReferenceMGASolver


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_reference_mga_calls_same_tensor_problem(backend):
    pytest.importorskip(backend)
    runtime = make_runtime(backend)
    initial = runtime.array(np.ones((1, 2)) * 0.5)
    problem = ControlProblem(
        lambda value: runtime.xp.sum(value * value), lambda value: value.reshape(-1) - 2
    )
    solver = ReferenceMGASolver(
        runtime,
        samples=8,
        seed=9,
        budget=RepairBudget(iterations=2, deadline_seconds=10),
    )
    result = solver.solve(problem, initial)
    assert result.predicted_feasible
    assert result.objective <= 0.5


@pytest.mark.parametrize("solver_type", [ModelMPPISolver, ReferenceMGASolver])
def test_nonfinite_predictions_never_become_feasible(solver_type):
    runtime = make_runtime("numpy")
    problem = ControlProblem(
        lambda value: runtime.xp.sum(value) * float("nan"),
        lambda value: value.reshape(-1) * float("nan"),
    )
    result = solver_type(runtime, samples=2, budget=RepairBudget(iterations=1)).solve(
        problem, np.zeros((1, 2))
    )
    assert not result.predicted_feasible and result.maximum_violation == float("inf")
