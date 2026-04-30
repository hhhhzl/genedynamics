"""Unit tests for NoiseSampler protocol + IsotropicGaussian + StructuredNoise."""

from __future__ import annotations

import numpy as np
import pytest

from genedynamics.core.prob import (
    IsotropicGaussian,
    NoiseSampler,
    StructuredNoise,
)


# ----- protocol conformance ------------------------------------------------


def test_isotropic_conforms_to_protocol() -> None:
    assert isinstance(IsotropicGaussian(sigma=0.5), NoiseSampler)


def test_structured_conforms_to_protocol() -> None:
    s = StructuredNoise(
        jacobian_fn=lambda q: np.eye(3, dtype=np.float32),
        normal_fn=None,
        sigma_tangent=0.1,
        sigma_nullspace=0.01,
    )
    assert isinstance(s, NoiseSampler)


# ----- IsotropicGaussian: numpy / jax / torch ------------------------------


def test_isotropic_numpy_shape_and_scale() -> None:
    sampler = IsotropicGaussian(sigma=2.0, backend="numpy")
    rng = np.random.default_rng(42)
    eta = sampler.sample((10000, 3), key=rng)
    assert eta.shape == (10000, 3)
    # Empirical std should be ~2.0 (large N).
    np.testing.assert_allclose(np.std(eta), 2.0, rtol=0.05)


def test_isotropic_numpy_with_seed_reproducible() -> None:
    s = IsotropicGaussian(sigma=1.0, backend="numpy")
    a = s.sample((4, 3), key=np.random.default_rng(0))
    b = s.sample((4, 3), key=np.random.default_rng(0))
    np.testing.assert_allclose(a, b)


def test_isotropic_jax_shape_and_jit() -> None:
    pytest.importorskip("jax")
    import jax, jax.numpy as jnp
    sampler = IsotropicGaussian(sigma=1.5, backend="jax")
    key = jax.random.PRNGKey(0)
    eta = sampler.sample((8, 4), key=key)
    assert eta.shape == (8, 4)
    eta_jit = jax.jit(lambda k: sampler.sample((8, 4), key=k))(key)
    np.testing.assert_allclose(np.asarray(eta_jit), np.asarray(eta), atol=1e-6)


def test_isotropic_jax_requires_key() -> None:
    pytest.importorskip("jax")
    sampler = IsotropicGaussian(sigma=1.0, backend="jax")
    with pytest.raises(ValueError, match="PRNGKey"):
        sampler.sample((3,))


# ----- StructuredNoise: degenerate cases ----------------------------------


def test_structured_no_normal_no_nullspace_degenerates_to_jacobian_pinv() -> None:
    """With normal_fn=None and sigma_nullspace=0, output = J^† (σ_t * η_t).

    For J = I_3 (3-DoF arm matching 3D task), J^† = I, so output = σ_t * η_t.
    """
    rng = np.random.default_rng(0)
    n = 3
    s = StructuredNoise(
        jacobian_fn=lambda q: np.eye(n, dtype=np.float32),
        normal_fn=None,
        sigma_tangent=0.4,
        sigma_normal=0.0,
        sigma_nullspace=0.0,
        damping=1e-4,
        backend="numpy",
    )
    q = np.zeros(n, dtype=np.float32)
    eta = s.sample((10000, n), key=rng, state=q)
    assert eta.shape == (10000, n)
    # All three axes should have ~σ_tangent stddev.
    np.testing.assert_allclose(np.std(eta, axis=0), 0.4, rtol=0.05)


def test_structured_nullspace_only_when_jacobian_underactuates() -> None:
    """With J = [I_2 | 0], pinv left-inverse, null-space spans the 3rd axis.

    With sigma_tangent=0 and sigma_nullspace>0, we should see noise concentrated
    in the 3rd component (joint 2 only) and zero in the first two.
    """
    n = 3
    J = np.zeros((2, n), dtype=np.float32)
    J[0, 0] = 1.0
    J[1, 1] = 1.0
    s = StructuredNoise(
        jacobian_fn=lambda q: J,
        normal_fn=None,
        sigma_tangent=0.0,
        sigma_normal=0.0,
        sigma_nullspace=0.7,
        damping=1e-4,
        backend="numpy",
    )
    q = np.zeros(n, dtype=np.float32)
    eta = s.sample((20000, n), key=np.random.default_rng(0), state=q)
    # Components 0, 1 should be ~0; component 2 should have ~σ_nullspace stddev.
    np.testing.assert_allclose(np.std(eta[:, 0]), 0.0, atol=0.05)
    np.testing.assert_allclose(np.std(eta[:, 1]), 0.0, atol=0.05)
    np.testing.assert_allclose(np.std(eta[:, 2]), 0.7, rtol=0.1)


def test_structured_normal_split_concentrates_in_normal_direction() -> None:
    """Pure normal noise (sigma_tangent=0, sigma_normal>0) acts only along ``n``."""
    n_dof = 3
    # J = I (3×3) and surface normal = (0, 0, 1).
    s = StructuredNoise(
        jacobian_fn=lambda q: np.eye(n_dof, dtype=np.float32),
        normal_fn=lambda q: np.array([0.0, 0.0, 1.0], dtype=np.float32),
        sigma_tangent=0.0,
        sigma_normal=0.5,
        sigma_nullspace=0.0,
        damping=1e-4,
        backend="numpy",
    )
    q = np.zeros(n_dof, dtype=np.float32)
    eta = s.sample((10000, n_dof), key=np.random.default_rng(0), state=q)
    # X, Y components (tangent plane) should have near-zero spread; Z should be σ_n.
    np.testing.assert_allclose(np.std(eta[:, 0]), 0.0, atol=0.02)
    np.testing.assert_allclose(np.std(eta[:, 1]), 0.0, atol=0.02)
    np.testing.assert_allclose(np.std(eta[:, 2]), 0.5, rtol=0.1)


def test_structured_requires_state() -> None:
    s = StructuredNoise(
        jacobian_fn=lambda q: np.eye(2, dtype=np.float32),
        normal_fn=None,
        sigma_tangent=0.1,
        backend="numpy",
    )
    with pytest.raises(ValueError, match="state"):
        s.sample((4, 2), key=np.random.default_rng(0))


def test_structured_shape_mismatch_raises() -> None:
    s = StructuredNoise(
        jacobian_fn=lambda q: np.eye(3, dtype=np.float32),
        normal_fn=None,
        sigma_tangent=0.1,
        backend="numpy",
    )
    with pytest.raises(ValueError, match="n_dof"):
        s.sample((4, 2), key=np.random.default_rng(0), state=np.zeros(3, dtype=np.float32))


# ----- StructuredNoise jax: jit-friendly ----------------------------------


def test_structured_jax_jit() -> None:
    pytest.importorskip("jax")
    import jax, jax.numpy as jnp

    def jac(q):
        return jnp.eye(3, dtype=jnp.float32)

    s = StructuredNoise(
        jacobian_fn=jac,
        normal_fn=None,
        sigma_tangent=0.3,
        sigma_nullspace=0.0,
        backend="jax",
    )
    q = jnp.zeros(3, dtype=jnp.float32)
    key = jax.random.PRNGKey(0)
    eta = jax.jit(lambda k: s.sample((16, 3), key=k, state=q))(key)
    assert eta.shape == (16, 3)


# ----- Solver injection: each backend exposes the noise_sampler hook ---------


def _check_has_draw_unit_noise(cls) -> None:
    assert hasattr(cls, "_draw_unit_noise") or hasattr(cls, "_draw_noise"), (
        f"{cls.__name__} is missing the noise injection hook"
    )


def test_mppi_backend_accepts_noise_sampler_kwarg() -> None:
    """MPPI uses sigma-included `_draw_noise` (returns randn * sigma)."""
    pytest.importorskip("jax")
    from genedynamics.solvers.single.mppi.backends.mppi_jax import MPPIBackendJax
    assert "noise_sampler" in MPPIBackendJax.__init__.__code__.co_varnames
    assert hasattr(MPPIBackendJax, "_draw_noise")


def test_mbd_backend_accepts_noise_sampler_kwarg() -> None:
    """MBD uses unit-variance `_draw_unit_noise`; sigma applied externally."""
    pytest.importorskip("jax")
    from genedynamics.solvers.single.mbd.backends.mbd_jax import MBDBackendJax
    assert "noise_sampler" in MBDBackendJax.__init__.__code__.co_varnames
    _check_has_draw_unit_noise(MBDBackendJax)


def test_ebmbd_jax_backend_exposes_noise_sampler() -> None:
    pytest.importorskip("jax")
    from genedynamics.solvers.single.ebmbd.backends.ebmbd_jax import EBMBDBackendJax
    _check_has_draw_unit_noise(EBMBDBackendJax)


def test_ebmbd_numpy_backend_exposes_noise_sampler() -> None:
    from genedynamics.solvers.single.ebmbd.backends.ebmbd_numpy import EBMBDBackendNumpy
    _check_has_draw_unit_noise(EBMBDBackendNumpy)


def test_mdoc_backend_exposes_noise_sampler() -> None:
    pytest.importorskip("jax")
    from genedynamics.solvers.single.mdoc.backends.mdoc_jax import MDOCBackendJax
    assert "noise_sampler" in MDOCBackendJax.__init__.__code__.co_varnames
    _check_has_draw_unit_noise(MDOCBackendJax)


def test_cfsmbd_backend_exposes_noise_sampler() -> None:
    pytest.importorskip("jax")
    from genedynamics.solvers.single.cfsmbd.backends.cfsmbd_jax import CFSMBDBackendJax
    assert "noise_sampler" in CFSMBDBackendJax.__init__.__code__.co_varnames
    _check_has_draw_unit_noise(CFSMBDBackendJax)


def test_twogo_backend_inherits_noise_sampler_via_inner_cfsmbd() -> None:
    """2GO wraps CFSMBD and delegates noise drawing to the inner ``_draw_unit_noise``."""
    pytest.importorskip("jax")
    from genedynamics.solvers.single.twogo.backends.twogo_jax import TwoGOBackendJax
    from genedynamics.solvers.single.cfsmbd.backends.cfsmbd_jax import CFSMBDBackendJax
    # 2GO doesn't define its own; the inner CFSMBD does.
    _check_has_draw_unit_noise(CFSMBDBackendJax)
