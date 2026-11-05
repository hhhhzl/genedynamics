"""A real constrained solve checks forward and derivative runtime semantics."""

import numpy as np
import pytest
from genedynamics.core.backends.differentiation import make_runtime
from genedynamics.solvers.single.budgeted_repair import (
    BudgetedRepairSolver,
    ControlProblem,
    RepairBudget,
)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_constraint_repair_and_derivatives(backend):
    runtime = make_runtime(backend)
    target = runtime.array([0.0, 0.0])
    problem = ControlProblem(
        lambda control: runtime.xp.sum((control - target) ** 2),
        lambda control: runtime.xp.stack((1.0 - control[0], -control[1] - 0.5)),
    )
    initial = runtime.array([0.0, -1.0])
    result = BudgetedRepairSolver(
        runtime, RepairBudget(iterations=25, deadline_seconds=30, trust_radius=1)
    ).solve(problem, initial)
    assert result.predicted_feasible
    assert result.maximum_violation <= 1e-4
    assert result.objective < 2
    assert runtime.to_numpy(result.controls)[0] >= 0.999
    _, gradient = runtime.value_and_grad(
        lambda control: runtime.xp.sum(control**2), runtime.array([2.0, -3.0])
    )
    np.testing.assert_allclose(runtime.to_numpy(gradient), [4, -6], atol=0.003)


def test_zero_iteration_budget_reports_infeasibility():
    runtime = make_runtime("numpy")
    problem = ControlProblem(
        lambda control: (control**2).sum(), lambda control: 1 - control
    )
    result = BudgetedRepairSolver(runtime, RepairBudget(iterations=0)).solve(
        problem, runtime.array([0.0])
    )
    assert not result.predicted_feasible
    assert result.iterations == 0
    assert result.termination_reason == "iteration_limit"
