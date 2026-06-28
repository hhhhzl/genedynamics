"""Stage-3 unit tests: budgeted adaptive-fidelity primitives.

Covers the pure functions behind the per-step fidelity selection in the method
scan: the value-of-information V_hat_k(ell) (eq:fidelity_value_estimator) and the
compute-budget dual update (eq:dual_update). The end-to-end scan (per-step ell*,
nu dynamics) is exercised by the GPU smoke; these lock the math.
"""
from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
from genedynamics.solvers.single.mrmfmbd.backends.mrmfmbd_mbd_jax import (  # noqa: E402
    compute_vhat, compute_vhat_rank, dual_update_nu,
)


def _rank_args():
    # a1,a2,a3,a4, logC, beta, rank_thresh, kappa
    return (1.0, 1.0, 0.5, 1.0, float(np.log(3)), 8.0, 0.85, 10.0)


def test_vhat_rank_fine_is_self_consistent():
    # The fine level (last) orders candidates like itself -> highest V_hat (no gate penalty),
    # while a rank-SCRAMBLING coarse level is gated out (V_hat << fine).
    rng = np.random.RandomState(0)
    true = rng.randn(8).astype(np.float32)
    scramble = rng.randn(8).astype(np.float32)
    faithful = true + 0.05 * rng.randn(8).astype(np.float32)
    rho = jnp.stack([jnp.asarray(scramble), jnp.asarray(faithful), jnp.asarray(true)], 0)  # (3,8)
    q = jnp.ones((3, 8, 1), jnp.float32)
    v = np.asarray(compute_vhat_rank(rho, q, jnp.float32(0.3), *_rank_args()))
    assert v.shape == (3,) and np.all(np.isfinite(v))
    assert v[0] < v[2], "rank-scrambling coarse level must be gated below fine"
    assert v[1] > v[0], "rank-faithful coarse level must beat the scrambler"


def test_vhat_rank_shift_invariant():
    # Denoise uses only candidate ordering; adding a constant to a level's scores (a softmax-inert
    # mean shift) must leave its rank-fidelity / V_hat unchanged -- the bug the old mis_l term had.
    rng = np.random.RandomState(1)
    true = rng.randn(8).astype(np.float32)
    coarse = true + 0.1 * rng.randn(8).astype(np.float32)
    rho = jnp.stack([jnp.asarray(coarse), jnp.asarray(true)], 0)
    q = jnp.ones((2, 8, 1), jnp.float32)
    args = (0.0, 0.0, 0.5, 0.0, float(np.log(2)), 8.0, 0.85, 10.0)  # a3-only (rank term) isolated
    v0 = np.asarray(compute_vhat_rank(rho, q, jnp.float32(0.3), *args))
    rho_shift = rho.at[0].add(7.0)
    v1 = np.asarray(compute_vhat_rank(rho_shift, q, jnp.float32(0.3), *args))
    assert abs(float(v1[0]) - float(v0[0])) < 1e-3, "constant shift must not change rank-fidelity"


def test_dual_update_nonneg_and_direction():
    nu = 0.5
    up = float(dual_update_nu(jnp.float32(nu), jnp.float32(1.0), jnp.float32(0.5), jnp.float32(0.1)))
    assert up > nu, "over-budget (C_star > Cbar) must raise nu"
    down = float(dual_update_nu(jnp.float32(nu), jnp.float32(0.0), jnp.float32(0.5), jnp.float32(0.1)))
    assert 0.0 <= down < nu, "under-budget must lower nu"
    floor = float(dual_update_nu(jnp.float32(0.0), jnp.float32(0.0), jnp.float32(1.0), jnp.float32(1.0)))
    assert floor == 0.0, "nu is floored at 0 (relu)"


def test_vhat_shape_and_finite():
    L, M, C = 3, 8, 4
    rng = np.random.default_rng(0)
    rho = jnp.asarray(rng.normal(size=(L, M)).astype("float32"))
    q = np.abs(rng.normal(size=(L, M, C))).astype("float32")
    q = jnp.asarray(q / q.sum(-1, keepdims=True))
    v = compute_vhat(rho, q, jnp.float32(0.5), 1.0, 1.0, 0.5, 1.0, float(np.log(C)))
    assert v.shape == (L,)
    assert np.all(np.isfinite(np.asarray(v)))


def test_vhat_variance_term_coupling():
    # The level whose candidates disagree more gets a larger variance contribution
    # (state-coupling: V_hat depends on the actual candidate batch).
    M, C = 8, 2
    rho = jnp.asarray(np.stack([np.zeros(M), np.linspace(-3, 3, M)]).astype("float32"))
    q = jnp.ones((2, M, C), "float32") / C
    v = compute_vhat(rho, q, jnp.float32(1.0), 1.0, 0.0, 0.0, 0.0, float(np.log(C)))  # var term only
    assert float(v[1]) > float(v[0])


def test_vhat_entropy_term_coupling():
    # Higher regime-posterior entropy -> larger entropy contribution.
    M, C = 6, 4
    rho = jnp.zeros((2, M), "float32")
    q_sharp = np.zeros((M, C), "float32"); q_sharp[:, 0] = 1.0          # zero entropy
    q_unif = np.ones((M, C), "float32") / C                            # max entropy
    q = jnp.asarray(np.stack([q_sharp, q_unif]))
    v = compute_vhat(rho, q, jnp.float32(1.0), 0.0, 1.0, 0.0, 0.0, float(np.log(C)))  # ent term only
    assert float(v[1]) > float(v[0])


def test_ellstar_cost_coupling():
    # ell* = argmax(V_hat - nu*C): with nu=0 pick the higher-VOI level; as nu
    # grows the cost penalty flips the choice to the cheaper level (the dual at
    # work — exactly what a fixed coarse->fine ladder cannot express).
    vhat = jnp.asarray([1.0, 1.05], "float32")   # level 1 marginally higher VOI
    C = jnp.asarray([0.2, 1.0], "float32")        # but level 1 is 5x costlier
    assert int(jnp.argmax(vhat - 0.0 * C)) == 1
    assert int(jnp.argmax(vhat - 0.5 * C)) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
