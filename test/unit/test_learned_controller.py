"""Stage-5a/5b unit tests: learned closed-loop controller primitives.

Covers the policy / morphology-embedding / time-encoding building blocks and the
gradient-traceability of the actuation w.r.t. the controller latent c (the
property the in-loop SHAC refinement relies on). The full closed-loop rollout +
in-loop SHAC are exercised by the GPU smoke.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from genedynamics.solvers.single.mrmfmbd.controller_system import (  # noqa: E402
    init_policy, policy_apply, init_embed, embed, time_encoding, obs_dim,
)


def test_obs_dim_and_policy_shapes():
    d_c, d_e, n_act = 8, 6, 10
    d_obs = obs_dim(d_c, d_e)
    assert d_obs == 1 + 2 + d_e + d_c
    pp = init_policy(0, d_obs, 32, n_act)
    obs = jnp.zeros((d_obs,), jnp.float32)
    a = policy_apply(pp, obs)
    assert a.shape == (n_act,)
    assert bool(jnp.all(jnp.abs(a) <= 1.0))            # tanh-bounded


def test_embedding_shape():
    E = init_embed(0, 27, 6)
    occ = jnp.ones((27,), jnp.float32) * 0.6
    e = embed(occ, E)
    assert e.shape == (6,) and bool(jnp.all(jnp.abs(e) <= 1.0))


def test_time_encoding_periodic():
    psi0 = np.asarray(time_encoding(0, 200))
    psiT = np.asarray(time_encoding(200, 200))           # one full period
    assert np.allclose(psi0, psiT, atol=1e-5)
    assert psi0.shape == (2,)


def test_actuation_grad_wrt_c_finite_nonzero():
    from genedynamics.envs.external.jax_mpm.scene import compute_actuation_learned

    class Cfg:
        env_horizon = 200
        feedback_v_scale = 1.0

    d_c, d_e, n_act = 8, 6, 10
    pp = init_policy(0, obs_dim(d_c, d_e), 32, n_act)
    ex = embed(jnp.ones((27,), jnp.float32) * 0.6, init_embed(0, 27, d_e))
    c = jnp.zeros((d_c,), jnp.float32)
    g = jax.grad(lambda cc: jnp.sum(
        compute_actuation_learned(cc, ex, jnp.int32(5), Cfg(), pp)))(c)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.sum(jnp.abs(g))) > 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
