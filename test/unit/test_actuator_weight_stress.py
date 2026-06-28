"""Stage 1 — continuous actuator_weight eigen-stress equivalence.

Kernel-level unit tests (NO MPM rollout, CPU-only) verifying the new
continuous-actuator eigen-stress `_stress_and_J_weighted` against the legacy
hard-index `_stress_and_J`:

  1. A one-hot weight (1 at the particle's actuator group, 0 elsewhere;
     all-zero for a passive particle) reproduces the legacy stress EXACTLY.
  2. A genuinely continuous weight matches the documented weighted sum
     A = Σ_i w_i · act_t[i] · (d_i ⊗ d_i).

This is the back-compat guarantee for Stage 1: as long as the simulator is
fed the one-hot derived from `actuator_id`, behavior is byte-identical to the
pre-extension code path. Full-rollout equivalence is a separate GPU check.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from genedynamics.envs.external.jax_mpm.scene import (  # noqa: E402
    MPMConfig,
    _stress_and_J,
    _stress_and_J_weighted,
)


def _onehot(aid: int, K: int) -> jnp.ndarray:
    w = np.zeros((K,), dtype=np.float32)
    if aid >= 0:
        w[aid] = 1.0
    return jnp.asarray(w)


def _synthetic(K: int = 10, seed: int = 0):
    """Synthetic per-particle stress inputs: small deformation, random
    activations and (unit) muscle directions."""
    rng = np.random.default_rng(seed)
    F = jnp.asarray(np.eye(3) + 0.05 * rng.standard_normal((3, 3)), dtype=jnp.float32)
    E = jnp.asarray(1.3, dtype=jnp.float32)
    act_t = jnp.asarray(rng.standard_normal(K).astype(np.float32))
    dirs = rng.standard_normal((K, 3)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
    return F, E, act_t, jnp.asarray(dirs)


@pytest.mark.parametrize("aid", [-1, 0, 1, 5, 9])
def test_onehot_matches_hard_index(aid):
    """One-hot(actuator_id) weighted stress == legacy hard-index stress."""
    cfg = MPMConfig()
    K = cfg.n_actuators
    F, E, act_t, muscle_dirs = _synthetic(K)

    tau_hard, J_hard = _stress_and_J(
        F, E, jnp.asarray(aid, jnp.int32), act_t, muscle_dirs, cfg
    )
    tau_w, J_w = _stress_and_J_weighted(
        F, E, _onehot(aid, K), act_t, muscle_dirs, cfg
    )

    np.testing.assert_allclose(np.asarray(tau_w), np.asarray(tau_hard), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(J_w), np.asarray(J_hard), rtol=1e-6)


def test_passive_row_is_zero_eigenstress():
    """All-zero weight (passive) == legacy actuator_id == -1 (no eigen-stress)."""
    cfg = MPMConfig()
    K = cfg.n_actuators
    F, E, act_t, muscle_dirs = _synthetic(K, seed=1)

    tau_passive, _ = _stress_and_J_weighted(
        F, E, jnp.zeros((K,), dtype=jnp.float32), act_t, muscle_dirs, cfg
    )
    tau_hard_passive, _ = _stress_and_J(
        F, E, jnp.asarray(-1, jnp.int32), act_t, muscle_dirs, cfg
    )
    np.testing.assert_allclose(
        np.asarray(tau_passive), np.asarray(tau_hard_passive), rtol=1e-5, atol=1e-6
    )


def test_continuous_blend_matches_manual_sum():
    """A soft weight matches the manual A = Σ_i w_i act_i d_i d_i^T sum."""
    cfg = MPMConfig()
    K = cfg.n_actuators
    F, E, act_t, muscle_dirs = _synthetic(K, seed=3)

    w = np.zeros((K,), dtype=np.float32)
    w[0], w[1] = 0.5, 0.25
    tau, _ = _stress_and_J_weighted(F, E, jnp.asarray(w), act_t, muscle_dirs, cfg)

    d = np.asarray(muscle_dirs)
    a = np.asarray(act_t)
    A = sum(w[i] * a[i] * np.outer(d[i], d[i]) for i in range(K))
    Fn = np.asarray(F)
    J = max(float(np.linalg.det(Fn)), cfg.j_min)
    mu = la = float(E) * cfg.scale
    B = Fn @ Fn.T
    tau_manual = (
        mu * (B - np.eye(3)) + la * np.log(J) * np.eye(3) + Fn @ A @ Fn.T
    )
    np.testing.assert_allclose(np.asarray(tau), tau_manual, rtol=1e-4, atol=1e-5)
