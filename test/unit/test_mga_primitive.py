"""Gates for the UPSTREAM position-stiffness primitive (core/control/stiffness):
svec round-trip, K=expm SPD+grad, action_size. Runs on fedguide CPU (no mjx).

NOTE: this primitive is the ONE genuinely-new MGA piece and lives UPSTREAM in
genedynamics/core/control/ (not solver-private) so any env/solver can use it."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from genedynamics.core.control.stiffness import (
    PrimitiveSpec,
    metric_GK_diag,
    metric_GK_diag_full,
    stiffness_log_to_pd,
    stiffness_pd_to_log,
    svec2sym,
    svec_len,
    sym2svec,
    unpack_primitive,
)


@pytest.mark.parametrize("d", [3, 6])
def test_svec_roundtrip(d):
    rng = np.random.default_rng(d)
    for _ in range(20):
        A = jnp.asarray(rng.standard_normal((d, d)), jnp.float64)
        M = 0.5 * (A + A.T)                       # symmetric
        v = sym2svec(M)
        assert v.shape == (svec_len(d),)
        Mr = svec2sym(v, d)
        assert jnp.allclose(Mr, M, atol=1e-12), jnp.max(jnp.abs(Mr - M))


@pytest.mark.parametrize("d", [3, 6])
def test_svec_batched(d):
    rng = np.random.default_rng(d + 1)
    A = jnp.asarray(rng.standard_normal((5, d, d)), jnp.float64)
    M = 0.5 * (A + jnp.swapaxes(A, -1, -2))
    assert jnp.allclose(svec2sym(sym2svec(M), d), M, atol=1e-12)


@pytest.mark.parametrize("d", [3, 6])
def test_stiffness_spd(d):
    rng = np.random.default_rng(100 + d)
    s = jnp.asarray(rng.standard_normal((svec_len(d),)), jnp.float64)
    K = stiffness_log_to_pd(s, d)
    assert jnp.allclose(K, K.T, atol=1e-10)      # symmetric
    lam = jnp.linalg.eigvalsh(K)
    assert float(lam.min()) > 0.0                # positive definite
    # round-trip log <-> pd
    s_back = stiffness_pd_to_log(K)
    assert jnp.allclose(stiffness_log_to_pd(s_back, d), K, atol=1e-8)


def test_stiffness_grad_finite():
    d = 3
    rng = np.random.default_rng(7)
    s = jnp.asarray(rng.standard_normal((svec_len(d),)), jnp.float64)
    g = jax.grad(lambda v: jnp.sum(stiffness_log_to_pd(v, d)))(s)
    assert jnp.all(jnp.isfinite(g))


def test_action_size_layout():
    # impedance primitive: 3 pos + svec(3)=6 stiffness + 2 feedforward = 11
    spec = PrimitiveSpec(pos_dim=3, stiff_dim=3, feed_dim=2)
    assert spec.stiff_width == 6
    assert spec.total_width == 11
    u = jnp.arange(spec.total_width, dtype=jnp.float32)
    r, s_vec, nu = unpack_primitive(u, spec)
    assert r.shape == (3,) and s_vec.shape == (6,) and nu.shape == (2,)
    assert jnp.allclose(r, u[:3]) and jnp.allclose(nu, u[9:11])
    # torque/position env (no stiffness) is unchanged in width
    flat = PrimitiveSpec(pos_dim=0, stiff_dim=0, feed_dim=12)
    assert flat.total_width == 12 and flat.stiff_width == 0


def test_GK_only_moves_stiffness_coords():
    spec = PrimitiveSpec(pos_dim=3, stiff_dim=3, feed_dim=2)
    diag = metric_GK_diag(spec, w_S=2.5)
    assert diag.shape == (11,)
    # nonzero ONLY on the stiffness svec block [3:9)
    assert jnp.allclose(diag[:3], 0.0) and jnp.allclose(diag[9:], 0.0)
    assert jnp.allclose(diag[3:9], 2.5)
    full = metric_GK_diag_full(spec, 2.5, n_steps=4)
    assert full.shape == (44,)
    assert float(jnp.sum(full > 0)) == 6 * 4     # 6 stiffness coords per step, 4 steps
