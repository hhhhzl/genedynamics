"""Byte-parity tests for the DIAL node<->dense spline (runs on fedguide x86).

Verifies the cached interpolation matrices reproduce jax_cosmo's k=2
InterpolatedUnivariateSpline (the exact spline dial-mpc uses) to rtol 1e-5, and
that the receding-horizon node shift matches dial-mpc's ``MBDPI.shift`` logic.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline as IUS

from genedynamics.solvers.single.dial.spline import NodeSpline


# dial-mpc defaults: Hnode=4, Hsample=16
HNODE, HSAMPLE, CTRL_DT = 4, 16, 0.02


@pytest.fixture(scope="module")
def spl():
    return NodeSpline.build(HNODE, HSAMPLE, CTRL_DT)


def _grids():
    span = CTRL_DT * HSAMPLE
    step_us = jnp.linspace(0.0, span, HSAMPLE + 1)
    step_nodes = jnp.linspace(0.0, span, HNODE + 1)
    return step_nodes, step_us


def test_matrix_shapes(spl):
    assert spl.N2U.shape == (HSAMPLE + 1, HNODE + 1)
    assert spl.U2N.shape == (HNODE + 1, HSAMPLE + 1)


def test_node2u_parity_vs_jax_cosmo(spl):
    # node2u(nodes) must equal jax_cosmo IUS(step_nodes, nodes, k=2)(step_us)
    step_nodes, step_us = _grids()
    rng = np.random.default_rng(0)
    for _ in range(20):
        nodes = jnp.asarray(rng.standard_normal((HNODE + 1,)), dtype=jnp.float32)
        ref = IUS(step_nodes, nodes, k=2)(step_us)
        got = spl.N2U @ nodes
        assert jnp.allclose(got, ref, rtol=1e-5, atol=1e-5), jnp.max(jnp.abs(got - ref))


def test_u2node_parity_vs_jax_cosmo(spl):
    step_nodes, step_us = _grids()
    rng = np.random.default_rng(1)
    for _ in range(20):
        us = jnp.asarray(rng.standard_normal((HSAMPLE + 1,)), dtype=jnp.float32)
        ref = IUS(step_us, us, k=2)(step_nodes)
        got = spl.U2N @ us
        assert jnp.allclose(got, ref, rtol=1e-5, atol=1e-5), jnp.max(jnp.abs(got - ref))


def test_node2u_multidim_matches_per_dim(spl):
    # (Hnode+1, nu) -> (Hsample+1, nu): each column independently spline-interp'd
    step_nodes, step_us = _grids()
    nu = 3
    rng = np.random.default_rng(2)
    nodes = jnp.asarray(rng.standard_normal((HNODE + 1, nu)), dtype=jnp.float32)
    got = spl.node2u(nodes)
    assert got.shape == (HSAMPLE + 1, nu)
    for j in range(nu):
        ref = IUS(step_nodes, nodes[:, j], k=2)(step_us)
        assert jnp.allclose(got[:, j], ref, rtol=1e-5, atol=1e-5)


def test_node2u_batch_matches_loop(spl):
    rng = np.random.default_rng(3)
    B, nu = 8, 2
    nodes_b = jnp.asarray(rng.standard_normal((B, HNODE + 1, nu)), dtype=jnp.float32)
    got = spl.node2u_batch(nodes_b)
    assert got.shape == (B, HSAMPLE + 1, nu)
    for b in range(B):
        assert jnp.allclose(got[b], spl.node2u(nodes_b[b]), rtol=1e-5, atol=1e-5)


def test_shift_matches_dial_logic(spl):
    # Reproduce dial-mpc MBDPI.shift: node2u -> roll -1 -> zero tail -> u2node
    rng = np.random.default_rng(4)
    nu = 2
    nodes = jnp.asarray(rng.standard_normal((HNODE + 1, nu)), dtype=jnp.float32)

    us = spl.node2u(nodes)
    us_rolled = jnp.roll(us, -1, axis=0).at[-1].set(jnp.zeros(nu))
    ref = spl.u2node(us_rolled)

    got = spl.shift_nodes(nodes)
    assert jnp.allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_terminal_hold_shift_preserves_planned_tail(spl):
    rng = np.random.default_rng(40)
    nodes = jnp.asarray(
        rng.standard_normal((HNODE + 1, 2)), dtype=jnp.float32
    )
    us = spl.node2u(nodes)
    ref = spl.u2node(
        jnp.roll(us, -1, axis=0).at[-1].set(us[-1])
    )

    got = spl.shift_nodes_terminal_hold(nodes)
    assert jnp.allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_node2u_is_linear(spl):
    # Sanity: the map is linear (so the cached-matrix premise holds).
    rng = np.random.default_rng(5)
    a = jnp.asarray(rng.standard_normal((HNODE + 1, 2)), dtype=jnp.float32)
    b = jnp.asarray(rng.standard_normal((HNODE + 1, 2)), dtype=jnp.float32)
    lhs = spl.node2u(2.0 * a - 3.0 * b)
    rhs = 2.0 * spl.node2u(a) - 3.0 * spl.node2u(b)
    assert jnp.allclose(lhs, rhs, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
