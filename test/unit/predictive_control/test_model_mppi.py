import numpy as np
import pytest
from genedynamics.core.backends.differentiation import make_runtime
from genedynamics.solvers.single.model_mppi import ModelMPPISolver
from genedynamics.solvers.single.budgeted_repair import ControlProblem, RepairBudget


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_mppi_uses_shared_objective_and_constraints(backend):
    runtime = make_runtime(backend)
    problem = ControlProblem(lambda control: runtime.xp.sum((control - 0.5) ** 2),
                             lambda control: -control.reshape(-1),
                             runtime.array(-1.0), runtime.array(1.0))
    initial = runtime.array(np.zeros((2, 1), np.float32))
    solver = ModelMPPISolver(runtime, samples=24, noise_scale=0.4,
                            budget=RepairBudget(iterations=3, deadline_seconds=30))
    result = solver.solve(problem, initial)
    assert result.objective < 0.2
    assert result.predicted_feasible
    assert result.model_evaluations == 72
