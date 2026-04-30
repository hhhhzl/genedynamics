"""Unit tests for CoverageMap."""

from __future__ import annotations

import numpy as np
import pytest

from genedynamics.core.coverage import CoverageMap


# ---- helpers --------------------------------------------------------------


def _grid_samples(n_per_axis: int = 5, span: float = 1.0) -> np.ndarray:
    """A small 2D-in-3D grid of n^2 samples on the z=0 plane."""
    xs = np.linspace(0.0, span, n_per_axis, dtype=np.float32)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())], axis=-1)


# ---- init ----------------------------------------------------------------


def test_init_state_is_zeros() -> None:
    s = _grid_samples(4)
    m = CoverageMap(samples=s)
    vs = m.init_state()
    assert vs.shape == (16,)
    np.testing.assert_array_equal(vs, np.zeros(16, dtype=np.float32))


# ---- update at exact sample point -----------------------------------------


def test_update_at_sample_point_strength_one() -> None:
    s = _grid_samples(3, span=1.0)        # 9 samples
    m = CoverageMap(samples=s, sigma=0.05)
    vs = m.update(m.init_state(), s[4])   # centre sample
    assert float(vs[4]) == pytest.approx(1.0, abs=1e-6)
    # Far corners decay; with σ=0.05 and centre-corner ≈ 0.707, kernel ≈ 0.
    assert float(vs[0]) < 1e-6


# ---- monotone (max-aggregation) -------------------------------------------


def test_update_is_monotone_along_path() -> None:
    s = _grid_samples(5, span=1.0)
    m = CoverageMap(samples=s, sigma=0.1)
    vs = m.init_state()
    path = np.linspace([0, 0, 0], [1, 1, 0], 20, dtype=np.float32)
    last = vs.copy()
    for p in path:
        vs = m.update(vs, p)
        assert np.all(vs >= last - 1e-6)
        last = vs.copy()


def test_update_idempotent_at_same_point() -> None:
    s = _grid_samples(3)
    m = CoverageMap(samples=s, sigma=0.1)
    p = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    vs = m.update(m.init_state(), p)
    vs2 = m.update(vs, p)
    np.testing.assert_allclose(vs, vs2, atol=1e-6)


def test_update_far_point_no_change() -> None:
    s = _grid_samples(3, span=1.0)
    m = CoverageMap(samples=s, sigma=0.05)
    vs = m.update(m.init_state(), s[4])
    vs2 = m.update(vs, np.array([100.0, 100.0, 100.0], dtype=np.float32))
    # Kernel from a 100-unit-away point with σ=0.05 is e^{-2e6} ≈ 0.
    np.testing.assert_allclose(vs, vs2, atol=1e-6)


# ---- coverage metrics -----------------------------------------------------


def test_coverage_mean_endpoints() -> None:
    s = _grid_samples(4)
    m = CoverageMap(samples=s)
    vs0 = m.init_state()
    vs1 = np.ones(16, dtype=np.float32)
    assert float(m.coverage_mean(vs0)) == 0.0
    assert float(m.coverage_mean(vs1)) == 1.0


def test_coverage_fraction_threshold() -> None:
    s = _grid_samples(4)
    m = CoverageMap(samples=s, threshold=0.5)
    vs = np.array([0.6, 0.4, 0.6, 0.4] * 4, dtype=np.float32)
    assert float(m.coverage_fraction(vs)) == pytest.approx(0.5)
    # Threshold parametrisation works.
    m2 = CoverageMap(samples=s, threshold=0.3)
    assert float(m2.coverage_fraction(vs)) == pytest.approx(1.0)


# ---- backend equivalence: jax matches numpy -------------------------------


def test_jax_backend_matches_numpy() -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp
    s_np = _grid_samples(5, span=1.0)
    s_jx = jnp.asarray(s_np)
    m_np = CoverageMap(samples=s_np, sigma=0.1, backend="numpy")
    m_jx = CoverageMap(samples=s_jx, sigma=0.1, backend="jax")
    vs_np = m_np.init_state()
    vs_jx = m_jx.init_state()
    path = np.array([[0.1, 0.1, 0.0], [0.5, 0.5, 0.0], [0.9, 0.9, 0.0]], dtype=np.float32)
    for p in path:
        vs_np = m_np.update(vs_np, p)
        vs_jx = m_jx.update(vs_jx, jnp.asarray(p))
    np.testing.assert_allclose(np.asarray(vs_jx), vs_np, atol=1e-6)
    np.testing.assert_allclose(
        float(m_jx.coverage_mean(vs_jx)),
        float(m_np.coverage_mean(vs_np)),
        atol=1e-6,
    )


# ---- jit / vmap / grad ----------------------------------------------------


def test_update_under_jit() -> None:
    pytest.importorskip("jax")
    import jax, jax.numpy as jnp
    s = jnp.asarray(_grid_samples(4))
    m = CoverageMap(samples=s, sigma=0.1, backend="jax")
    p = jnp.array([0.5, 0.5, 0.0], dtype=jnp.float32)
    vs0 = m.init_state()
    vs_jit = jax.jit(m.update)(vs0, p)
    vs_eager = m.update(vs0, p)
    np.testing.assert_allclose(np.asarray(vs_jit), np.asarray(vs_eager), atol=1e-6)


def test_vmap_independent_rollouts() -> None:
    """B independent rollouts each carrying their own visit_strength."""
    pytest.importorskip("jax")
    import jax, jax.numpy as jnp
    s = jnp.asarray(_grid_samples(4))
    m = CoverageMap(samples=s, sigma=0.1, backend="jax")
    B = 8
    vs_batch = jnp.zeros((B, s.shape[0]), dtype=s.dtype)
    p_batch = jnp.asarray(np.random.RandomState(0).uniform(0, 1, (B, 3))).astype(jnp.float32)
    p_batch = p_batch.at[:, 2].set(0.0)
    vs_new = jax.vmap(m.update)(vs_batch, p_batch)
    assert vs_new.shape == (B, s.shape[0])
    # Each row should have at least one cell with non-trivial strength
    # (max along N is the closest-sample's kernel value).
    assert bool(jnp.all(vs_new.max(axis=-1) > 0.0))


def test_scan_through_path_and_grad() -> None:
    """Reward = mean(visit_strength) at end of an N-step path; grad wrt path."""
    pytest.importorskip("jax")
    import jax, jax.numpy as jnp
    s = jnp.asarray(_grid_samples(5, span=1.0))
    m = CoverageMap(samples=s, sigma=0.1, backend="jax")

    def coverage_at_path_end(path):
        def step(vs, p):
            return m.update(vs, p), None
        vs_final, _ = jax.lax.scan(step, m.init_state(), path)
        return m.coverage_mean(vs_final)

    path = jnp.asarray(
        np.linspace([0, 0, 0], [1, 1, 0], 10, dtype=np.float32)
    )
    cov = coverage_at_path_end(path)
    assert float(cov) > 0.2 and float(cov) <= 1.0
    # Differentiable end-to-end (smooth max via a soft kernel).
    g = jax.grad(coverage_at_path_end)(path)
    assert g.shape == path.shape
    assert bool(jnp.all(jnp.isfinite(g)))


# ---- non-default sigma / threshold via constructor ------------------------


def test_sigma_parametrised() -> None:
    """Larger sigma → larger neighbourhood gets credit per visit."""
    s = _grid_samples(5)
    m_tight = CoverageMap(samples=s, sigma=0.02)
    m_wide = CoverageMap(samples=s, sigma=0.2)
    p = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    vs_tight = m_tight.update(m_tight.init_state(), p)
    vs_wide = m_wide.update(m_wide.init_state(), p)
    # Wider σ spreads a single visit across more cells.
    assert float(m_wide.coverage_mean(vs_wide)) > float(m_tight.coverage_mean(vs_tight))
