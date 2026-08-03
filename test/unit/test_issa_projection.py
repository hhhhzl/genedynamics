"""Task-adapted ISSA/AdamBA transition-safety tests."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from genedynamics.solvers.single.issa.backends.issa_jax import IssaProjection


class _State(NamedTuple):
    x: jnp.ndarray
    info: dict


class _IntervalEnv:
    """Safe set |x| <= 1; the first action component controls one-step x."""

    def safety_index(self, state):
        return jnp.abs(state.x) - 1.0

    def step(self, state, action):
        return _State(
            x=state.x + action[0],
            info={"step": state.info["step"] + 1},
        )


class _NoSafeActionEnv:
    """All candidates are unsafe; zero is the unique least-risk hold."""

    def safety_index(self, state):
        return state.x

    def step(self, state, action):
        return _State(
            x=1.0 + jnp.sum(action ** 2),
            info={"step": state.info["step"] + 1},
        )


def _state(x, step=0):
    return _State(jnp.asarray(x, jnp.float32), {"step": jnp.asarray(step, jnp.int32)})


def test_already_transition_safe_action_is_identity():
    projection = IssaProjection(_IntervalEnv(), n_dirs=32, n_iters=40, seed=4)
    state = _state(0.5)
    nominal = jnp.asarray([-0.2, 0.1], jnp.float32)
    action, info = projection.project_with_info(state, nominal)
    np.testing.assert_allclose(action, nominal, atol=0.0)
    assert not bool(info["failure"])
    assert float(info["intervention"]) == 0.0


def test_unsafe_action_is_projected_to_nonincreasing_next_safety():
    projection = IssaProjection(
        _IntervalEnv(), n_dirs=128, n_iters=50, bound=1e-4, seed=2
    )
    state = _state(0.9)
    nominal = jnp.asarray([0.5, 0.0], jnp.float32)
    action, info = projection.project_with_info(state, nominal)
    assert float(projection._transition_margin(state, action)) <= 1e-5
    assert float(action[0]) <= 0.0
    assert bool(info["found_safe"])
    assert float(info["intervention"]) > 0.0


def test_no_safe_ray_uses_minimum_risk_hold_and_reports_failure():
    projection = IssaProjection(
        _NoSafeActionEnv(), n_dirs=24, n_iters=25, seed=1
    )
    nominal = jnp.asarray([0.7, -0.4], jnp.float32)
    action, info = projection.project_with_info(_state(0.0), nominal)
    np.testing.assert_allclose(action, jnp.zeros_like(nominal), atol=1e-6)
    assert bool(info["failure"])
    assert not bool(info["found_safe"])


def test_projection_is_deterministic_for_seed_and_state_step():
    p1 = IssaProjection(_IntervalEnv(), n_dirs=64, n_iters=40, seed=9)
    p2 = IssaProjection(_IntervalEnv(), n_dirs=64, n_iters=40, seed=9)
    state = _state(0.9, step=7)
    nominal = jnp.asarray([0.5, 0.2], jnp.float32)
    a1, i1 = p1.project_with_info(state, nominal)
    a2, i2 = p2.project_with_info(state, nominal)
    np.testing.assert_allclose(a1, a2, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(i1["margin"], i2["margin"], rtol=0.0, atol=0.0)


def test_compiled_projection_matches_eager_contract():
    projection = IssaProjection(
        _IntervalEnv(), n_dirs=64, n_iters=40, bound=1e-4, seed=6
    )
    compiled = jax.jit(
        lambda state, action: projection.project_jax(state, action)
    )
    state = _state(0.9, step=3)

    for nominal in (
        jnp.asarray([-0.2, 0.1], jnp.float32),
        jnp.asarray([0.5, 0.2], jnp.float32),
    ):
        eager_action, eager_info = projection.project_with_info(state, nominal)
        compiled_action, compiled_info = compiled(state, nominal)
        np.testing.assert_allclose(compiled_action, eager_action, atol=1e-6)
        for key in ("found_safe", "failure", "margin", "intervention"):
            np.testing.assert_allclose(
                compiled_info[key], eager_info[key], atol=1e-6
            )
