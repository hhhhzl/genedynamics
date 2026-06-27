"""Proof-as-test: the MBD reverse update IS the DIAL weighted-mean update.

DIAL-MPC == (MBD-style control-space diffusion) + (receding-horizon MPC bridge).
This file pins the algebraic identity that justifies that statement: MBD's
DDPM score-form reverse step (mbd_jax.py:329-334, transport=None default)
reduces EXACTLY to DIAL's pure self-normalised weighted mean
(dial_core.py: ``Ybar = einsum(weights, Y0s)``) because the ``alpha_bar`` terms
cancel. So the bridge over the MBD weighted-mean planner reproduces DIAL's core;
the only remaining differences are config/wrapper knobs (node spline, per-node
geometric noise schedule, env-reward vs energy+guidance shaping, incumbent
baseline/pin/append, extra_sigma kick), NOT the update algorithm.
"""

import jax.numpy as jnp
import numpy as np
import pytest


def _mbd_ddpm_update(Ybar_curr, Ybar_w, abar_k, abar_km1, alpha_k):
    """Verbatim mbd_jax.py:329-334 default (transport=None) score-form update."""
    Yi = Ybar_curr * jnp.sqrt(abar_k)
    score = (-Yi + jnp.sqrt(abar_k) * Ybar_w) / (1.0 - abar_k)
    Yim1 = (Yi + (1.0 - abar_k) * score) / jnp.sqrt(alpha_k)
    return Yim1 / jnp.sqrt(abar_km1)


@pytest.mark.parametrize("idx", [1, 10, 25, 50, 75, 99])
def test_mbd_update_equals_weighted_mean(idx):
    rng = np.random.default_rng(idx)
    Ybar_curr = jnp.asarray(rng.standard_normal((5, 3)), dtype=jnp.float32)
    Ybar_w = jnp.asarray(rng.standard_normal((5, 3)), dtype=jnp.float32)

    betas = jnp.linspace(1e-4, 1e-2, 100)
    abar = jnp.cumprod(1.0 - betas)
    alphas = 1.0 - betas

    mbd = _mbd_ddpm_update(Ybar_curr, Ybar_w, abar[idx], abar[idx - 1], alphas[idx])
    dial = Ybar_w  # DIAL: Ybar_next = pure weighted mean

    assert jnp.allclose(mbd, dial, atol=1e-5), float(jnp.max(jnp.abs(mbd - dial)))


def test_independent_of_beta_schedule():
    # The identity holds for ANY alpha_bar chain (the cancellation is exact).
    rng = np.random.default_rng(7)
    Ybar_curr = jnp.asarray(rng.standard_normal((5, 2)), dtype=jnp.float32)
    Ybar_w = jnp.asarray(rng.standard_normal((5, 2)), dtype=jnp.float32)
    for b0, bT in [(1e-4, 1e-2), (1e-3, 0.2), (1e-5, 5e-2)]:
        abar = jnp.cumprod(1.0 - jnp.linspace(b0, bT, 50))
        alphas = 1.0 - jnp.linspace(b0, bT, 50)
        for idx in (1, 20, 49):
            out = _mbd_ddpm_update(Ybar_curr, Ybar_w, abar[idx], abar[idx - 1], alphas[idx])
            assert jnp.allclose(out, Ybar_w, atol=1e-5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
