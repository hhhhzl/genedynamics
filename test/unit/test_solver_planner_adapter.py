"""SolverPlannerAdapter: drive ANY solver through the receding-horizon bridge.

The bridge (``RecedingHorizonController``) is solver-agnostic but only DIAL
implemented its planner protocol natively. ``SolverPlannerAdapter`` supplies the
protocol for everyone else by re-solving each step -- so MBD / 2GO / CFS-MBD /
... run closed-loop through the SAME bridge WITHOUT editing the solver.

Flat-state single-integrator task (no brax/mjx) so this runs in the CPU gate.
"""

import numpy as np
import pytest

from genedynamics.envs.factories import make_env, make_energy
from genedynamics.solvers.common.receding_horizon import (
    RecedingHorizonController,
    SolverPlannerAdapter,
)


def _run(solver, env, n_steps=25, rng=None):
    adapter = SolverPlannerAdapter(solver, horizon=getattr(solver, "horizon", 20))
    ctrl = RecedingHorizonController(
        adapter, step_fn=env.transition, n_steps=n_steps,
        n_diffuse_init=1, n_diffuse=1,
    )
    x0 = np.array([0.8, 0.8], dtype=np.float32)  # away from origin target
    res = ctrl.run(x0, rng)
    return res, float(env.cost(res.states[0])), float(env.cost(res.states[-1]))


def test_adapter_satisfies_planner_protocol():
    from genedynamics.solvers.common.receding_horizon import WarmStartPlanner

    class _Solver:
        horizon = 5
        def solve(self, x0, horizon=5, rng_key=None):
            class T:  # constant push toward the origin
                actions = [np.array([-0.4, -0.4], dtype=np.float32)] * horizon
            return T()

    assert isinstance(SolverPlannerAdapter(_Solver()), WarmStartPlanner)


def test_mock_solver_runs_closed_loop_through_bridge():
    env = make_env("single_integrator_box_2d")

    class _Solver:
        horizon = 10
        def solve(self, x0, horizon=10, rng_key=None):
            class T:
                actions = [np.array([-0.4, -0.4], dtype=np.float32)] * horizon
            return T()

    res, c0, cT = _run(_Solver(), env, n_steps=12)
    assert len(res.actions) == 12          # bridge executed every step
    assert cT < c0                         # closed loop made progress toward target


@pytest.mark.parametrize("name", ["mbd", "2go"])
def test_real_solver_through_bridge(name):
    pytest.importorskip("jax")
    import jax
    from genedynamics.core.backends.runtime import RuntimeBackendManager

    RuntimeBackendManager.set_backend("jax", device="cpu")
    backend = RuntimeBackendManager.get_backend()
    env = make_env("single_integrator_box_2d")
    energy = make_energy("single_integrator_box_2d")

    if name == "mbd":
        from genedynamics.solvers import MBDSolver
        solver = MBDSolver(dynamics=env, energy=energy, backend=backend,
                           horizon=20, Nsample=64, Ndiffuse=6)
    else:
        from genedynamics.solvers import TwoGOSolver
        solver = TwoGOSolver(dynamics=env, energy=energy, backend=backend,
                             horizon=20, Nsample=64, Ndiffuse=6)

    res, c0, cT = _run(solver, env, n_steps=25, rng=jax.random.PRNGKey(0))
    assert len(res.actions) == 25
    assert cT < 0.5 * c0, f"{name}+bridge made no progress: {c0:.3f} -> {cT:.3f}"
