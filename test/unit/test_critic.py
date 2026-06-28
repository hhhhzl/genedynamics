"""Stage-5b unit tests: the in-loop TD(lambda) value critic V_psi.

Covers the value MLP, the lambda-return targets, and that one SGD step reduces
the TD loss (the critic actually learns) — the building blocks of the in-scan
online critic that bootstraps the SHAC short-horizon return.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from genedynamics.solvers.single.mrmfmbd.controller_system import (  # noqa: E402
    init_critic, critic_value, lambda_returns, critic_td_loss, critic_sgd_step,
    state_dim,
)


def test_state_dim_and_value_shape():
    d_e = 8
    d_s = state_dim(d_e)
    assert d_s == 6 + d_e
    p = init_critic(0, d_s, 16)
    s = jnp.zeros((5, d_s), jnp.float32)
    v = critic_value(p, s)
    assert v.shape == (5,)


def test_lambda_returns_lam1_is_montecarlo():
    # lam=1 -> full Monte-Carlo return sum_t' gamma^{t'-t} r_t' (+ bootstrap tail).
    r = jnp.asarray([1.0, 2.0, 3.0], jnp.float32)
    v_next = jnp.zeros(3, jnp.float32)              # zero bootstrap
    g = np.asarray(lambda_returns(r, v_next, gamma=0.5, lam=1.0))
    # G_2 = 3 ; G_1 = 2 + .5*3 = 3.5 ; G_0 = 1 + .5*3.5 = 2.75
    assert np.allclose(g, [2.75, 3.5, 3.0], atol=1e-5)


def test_lambda_returns_lam0_is_one_step_td():
    r = jnp.asarray([1.0, 2.0, 3.0], jnp.float32)
    v_next = jnp.asarray([10.0, 20.0, 30.0], jnp.float32)
    g = np.asarray(lambda_returns(r, v_next, gamma=0.5, lam=0.0))
    # G_t = r_t + gamma*v_next_t
    assert np.allclose(g, [1 + 0.5 * 10, 2 + 0.5 * 20, 3 + 0.5 * 30], atol=1e-5)


def test_sgd_step_reduces_td_loss():
    rng = np.random.default_rng(0)
    B, h, d_e = 4, 6, 8
    d_s = state_dim(d_e)
    states = jnp.asarray(rng.normal(size=(B, h, d_s)).astype("float32"))
    rewards = jnp.asarray(rng.normal(size=(B, h)).astype("float32"))
    p = init_critic(0, d_s, 32)
    l0 = float(critic_td_loss(p, states, rewards, 0.9, 0.95))
    for _ in range(20):
        p = critic_sgd_step(p, states, rewards, 0.9, 0.95, lr=1e-2)
    l1 = float(critic_td_loss(p, states, rewards, 0.9, 0.95))
    assert l1 < l0, f"critic should reduce TD loss ({l1} !< {l0})"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
