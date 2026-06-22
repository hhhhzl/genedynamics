"""Smoke tests for DIALMPCSolver (multi-backend wrapper) on CPU.

Constructs the solver with an injected env-reward rollout + real-step (no mjx),
selects the jax backend via RuntimeBackendManager, runs solve() (receding-horizon
bridge), and checks it returns a valid Trajectory and registers as 'dial'.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from genedynamics.solvers.single.dial import DIALMPCSolver
from genedynamics.core.types import Trajectory
from genedynamics.core.backends.runtime import RuntimeBackendManager

NU = 2
TARGET = 0.5


@pytest.fixture(autouse=True, scope="module")
def _jax_backend():
    RuntimeBackendManager.set_backend("jax")
    yield


class _MockBackend:
    name = "jax"
    device = "cpu"


class _MockDynamics:
    act_dim = NU

    def step(self, state, action):
        action = jnp.asarray(action)
        du = action.shape[0]
        pos, vel = state[:du], state[du:]
        vel = vel + action
        pos = pos + vel
        return jnp.concatenate([pos, vel])


def _rollout_fn(state, us, t0=0.0):
    return -jnp.sum((us - TARGET) ** 2, axis=-1)


def _step_fn(state, action):
    return _MockDynamics().step(state, action)


def _make_solver(**kw):
    return DIALMPCSolver(
        dynamics=_MockDynamics(),
        energy=None,
        backend=_MockBackend(),
        nu=NU,
        rollout_fn=_rollout_fn,
        step_fn=_step_fn,
        Nsample=256,
        temp_sample=0.1,
        Ndiffuse=4,
        Ndiffuse_init=25,
        **kw,
    )


def test_solve_returns_trajectory():
    solver = _make_solver()
    traj = solver.solve(jnp.zeros(4), horizon=5, rng_key=jax.random.PRNGKey(0))
    assert isinstance(traj, Trajectory)
    assert len(traj.actions) == 5
    assert len(traj.states) == 6
    assert traj.info["source"] == "dial_mpc"
    for a in traj.actions:
        assert np.asarray(a).shape == (NU,)
        assert np.all(np.isfinite(np.asarray(a)))


def test_solve_warms_up_to_optimum():
    solver = _make_solver()
    traj = solver.solve(jnp.zeros(4), horizon=6, rng_key=jax.random.PRNGKey(1))
    last = float(np.mean(np.asarray(traj.actions[-1])))
    assert abs(last - TARGET) < 0.15, last


def test_n_steps_override():
    solver = _make_solver(n_steps=3)
    traj = solver.solve(jnp.zeros(4), horizon=99, rng_key=jax.random.PRNGKey(2))
    assert len(traj.actions) == 3


def test_backend_is_unified_and_warmstart_capable():
    # _get_backend_impl returns the jax backend exposing BOTH plan() and the
    # WarmStartPlanner methods the bridge needs.
    solver = _make_solver()
    backend = solver._get_backend_impl()
    for m in ("plan", "init_plan_var", "replan", "first_action", "shift", "make_schedule"):
        assert hasattr(backend, m), m


def test_single_shot_plan_works():
    # The backend's unified plan() (open-loop full-horizon) also runs.
    solver = _make_solver()
    backend = solver._get_backend_impl()
    out = backend.plan(jnp.zeros(4), jax.random.PRNGKey(3))
    assert "actions" in out and "states" in out and "rewards" in out
    assert out["actions"].shape == (17, NU)  # Hsample+1


def test_registered_in_registry():
    from genedynamics.core.registry.solvers import get_solver_registry

    reg = get_solver_registry()
    assert "dial" in reg.list_available()
    assert reg.get_class("dial") is DIALMPCSolver


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
