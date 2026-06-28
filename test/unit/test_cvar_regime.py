"""Stage 4 (contribution 4) — CVaR / DRO regime posterior.

Synthetic-reward unit tests (CPU, NO simulator) for the adversarial CVaR_α
regime marginalizer that complements the existing entropic (KL-ball) risk
posterior. Verifies the worst-case limits and that the induced posterior
emphasizes failure-prone regimes.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from genedynamics.solvers.single.mrmfmbd.mode_system.regime_posterior import (  # noqa: E402
    cvar_marginalize_jax,
    risk_sensitive_marginalize_jax,
    validate_mode,
    REGIME_POSTERIOR_MODES,
)


def _uniform_logprior(C):
    return jnp.log(jnp.ones((C,), dtype=jnp.float32) / C)


def test_modes_include_cvar():
    assert "cvar" in REGIME_POSTERIOR_MODES
    assert validate_mode("cvar") == "cvar"


def test_alpha1_is_prior_mean():
    R = jnp.asarray([[1.0, 2.0, 3.0, 4.0]])
    rho, q = cvar_marginalize_jax(R, _uniform_logprior(4), jnp.asarray(1.0))
    np.testing.assert_allclose(np.asarray(rho), [2.5], atol=1e-5)
    np.testing.assert_allclose(np.asarray(q[0]), [0.25] * 4, atol=1e-5)


def test_alpha_small_is_worst_regime():
    R = jnp.asarray([[5.0, 1.0, 3.0, 4.0]])  # worst regime = idx 1
    rho, q = cvar_marginalize_jax(R, _uniform_logprior(4), jnp.asarray(1e-3))
    np.testing.assert_allclose(np.asarray(rho), [1.0], atol=1e-4)
    assert int(np.argmax(np.asarray(q[0]))) == 1
    np.testing.assert_allclose(float(np.asarray(q[0])[1]), 1.0, atol=1e-3)


def test_cvar_monotone_increasing_in_alpha():
    rng = np.random.default_rng(0)
    R = jnp.asarray(rng.standard_normal((8, 5)).astype(np.float32))
    lp = _uniform_logprior(5)
    rhos = [float(cvar_marginalize_jax(R, lp, jnp.asarray(a))[0].mean())
            for a in (0.05, 0.2, 0.5, 1.0)]
    assert all(rhos[i] <= rhos[i + 1] + 1e-5 for i in range(len(rhos) - 1))


def test_cvar_within_min_and_mean_and_q_normalized():
    rng = np.random.default_rng(1)
    R = jnp.asarray(rng.standard_normal((6, 4)).astype(np.float32))
    rho, q = cvar_marginalize_jax(R, _uniform_logprior(4), jnp.asarray(0.5))
    Rn = np.asarray(R)
    rho_n = np.asarray(rho)
    assert np.all(rho_n >= Rn.min(axis=1) - 1e-4)
    assert np.all(rho_n <= Rn.mean(axis=1) + 1e-4)
    np.testing.assert_allclose(np.asarray(q).sum(axis=1), np.ones(6), atol=1e-4)


def test_cvar_posterior_emphasizes_low_reward_regimes():
    R = jnp.asarray([[0.0, 10.0, 1.0, 9.0]])  # worst two = idx 0, 2
    _, q = cvar_marginalize_jax(R, _uniform_logprior(4), jnp.asarray(0.5))
    qn = np.asarray(q[0])
    assert qn[0] > 0 and qn[2] > 0
    assert qn[1] == 0.0 and qn[3] == 0.0


def test_cvar_and_entropic_agree_at_worst_case_limit():
    # α→0 (CVaR) and τ_r→0 (entropic) both collapse to the worst regime.
    R = jnp.asarray([[2.0, -1.0, 0.5]])
    lp = _uniform_logprior(3)
    rho_cvar, _ = cvar_marginalize_jax(R, lp, jnp.asarray(1e-4))
    rho_ent, _ = risk_sensitive_marginalize_jax(R, lp, jnp.asarray(1e-3))
    np.testing.assert_allclose(np.asarray(rho_cvar), np.asarray(rho_ent), atol=1e-2)
