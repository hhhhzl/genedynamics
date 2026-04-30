"""Unit tests for SDFGrid3D."""

from __future__ import annotations

import numpy as np
import pytest

from genedynamics.envs.obstacles import Obstacle
from genedynamics.envs.obstacles.sdf_grid_3d import SDFGrid3D


def _sphere_grid(N: int = 32, radius: float = 0.4) -> SDFGrid3D:
    """Build an analytic sphere SDF on a uniform [-1, 1]^3 grid."""
    side = 2.0
    spacing = side / (N - 1)
    xs = np.linspace(-1.0, 1.0, N, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    voxels = (np.sqrt(X ** 2 + Y ** 2 + Z ** 2) - radius).astype(np.float32)
    return SDFGrid3D(
        voxels=voxels,
        origin=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        spacing=np.array([spacing] * 3, dtype=np.float32),
    )


# --- protocol conformance --------------------------------------------------


def test_obstacle_protocol_conformance() -> None:
    grid = _sphere_grid()
    assert isinstance(grid, Obstacle)


def test_aabb_and_center() -> None:
    grid = _sphere_grid()
    np.testing.assert_allclose(grid.aabb_min, [-1.0, -1.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(grid.aabb_max, [1.0, 1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(grid.center, [0.0, 0.0, 0.0], atol=1e-6)


# --- SDF accuracy on sphere ------------------------------------------------


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_sphere_sdf_values(backend: str) -> None:
    grid = _sphere_grid()
    pts = np.array([
        [0.7, 0.0, 0.0],     # outside, true=0.3
        [0.1, 0.0, 0.0],     # inside,  true=-0.3
        [0.0, 0.0, 0.4],     # on,      true=0.0
    ], dtype=np.float32)
    expected = np.array([0.3, -0.3, 0.0], dtype=np.float32)
    if backend == "jax":
        import jax.numpy as jnp
        phi = np.asarray(grid.sdf(jnp.asarray(pts), backend="jax"))
    else:
        phi = np.asarray(grid.sdf(pts, backend="numpy"))
    # Expect ≤ ~1 voxel of trilinear discretization error (spacing ~ 0.064).
    np.testing.assert_allclose(phi, expected, atol=1.5e-2)


def test_numpy_jax_agree() -> None:
    """JAX backend must match numpy backend bit-close."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    grid = _sphere_grid()
    pts = np.random.RandomState(0).uniform(-0.8, 0.8, (32, 3)).astype(np.float32)
    phi_np, g_np = grid.sdf_and_grad(pts, backend="numpy")
    phi_j, g_j = grid.sdf_and_grad(jnp.asarray(pts), backend="jax")
    np.testing.assert_allclose(np.asarray(phi_j), phi_np, atol=1e-6)
    np.testing.assert_allclose(np.asarray(g_j), g_np, atol=1e-6)


# --- analytic gradient vs finite difference --------------------------------


def test_analytic_gradient_matches_fd() -> None:
    """Analytic trilinear grad should match a centred FD inside one voxel."""
    grid = _sphere_grid(N=64)
    pts = np.array([
        [0.55, 0.10, -0.20],
        [-0.30, 0.40, 0.05],
        [0.20, -0.45, 0.30],
    ], dtype=np.float32)
    _, g_an = grid.sdf_and_grad(pts, backend="numpy")
    g_fd = np.zeros_like(g_an)
    eps = 1e-3
    for i in range(3):
        d = np.zeros(3, dtype=np.float32); d[i] = eps
        phi_p = grid.sdf(pts + d, backend="numpy")
        phi_m = grid.sdf(pts - d, backend="numpy")
        g_fd[..., i] = (phi_p - phi_m) / (2 * eps)
    np.testing.assert_allclose(g_an, g_fd, atol=2e-3)


def test_gradient_unit_norm_outside_sphere() -> None:
    """For a pure sphere, ‖∇φ‖ ≈ 1 outside (analytic SDF property)."""
    grid = _sphere_grid(N=64)
    # Pick points well outside the sphere, away from the AABB boundary.
    pts = np.array([[0.7, 0.0, 0.0], [0.0, 0.6, 0.0], [0.0, 0.0, -0.7]],
                   dtype=np.float32)
    _, g = grid.sdf_and_grad(pts, backend="numpy")
    norms = np.linalg.norm(g, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=2e-2)


# --- jit / vmap / grad through ---------------------------------------------


def test_jit_sdf() -> None:
    pytest.importorskip("jax")
    import jax, jax.numpy as jnp
    grid = _sphere_grid()
    fn = jax.jit(lambda p: grid.sdf(p, backend="jax"))
    p = jnp.array([0.5, 0.0, 0.0], dtype=jnp.float32)
    np.testing.assert_allclose(
        float(fn(p)), float(grid.sdf(np.asarray(p), backend="numpy")), atol=1e-5
    )


def test_vmap_sdf_batch() -> None:
    pytest.importorskip("jax")
    import jax, jax.numpy as jnp
    grid = _sphere_grid()
    pts = jnp.array(
        np.random.RandomState(0).uniform(-0.8, 0.8, (16, 3)), dtype=jnp.float32
    )
    phi = jax.vmap(lambda p: grid.sdf(p, backend="jax"))(pts)
    assert phi.shape == (16,)


def test_jax_grad_through_sdf() -> None:
    """sdf_and_grad's analytic ∇φ must agree with jax.grad of sdf()."""
    pytest.importorskip("jax")
    import jax, jax.numpy as jnp
    grid = _sphere_grid(N=64)
    p = jnp.array([0.5, 0.1, -0.2], dtype=jnp.float32)
    g_grad = jax.grad(lambda q: grid.sdf(q, backend="jax"))(p)
    _, g_an = grid.sdf_and_grad(p, backend="jax")
    np.testing.assert_allclose(np.asarray(g_grad), np.asarray(g_an), atol=1e-5)


# --- project ---------------------------------------------------------------


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_project_to_zero_level_set(backend: str) -> None:
    grid = _sphere_grid(N=96)  # finer grid → tighter projection
    rng = np.random.RandomState(0)
    # Random unit directions, perturbed at radius 0.55 (away from sphere r=0.4).
    dirs = rng.randn(64, 3); dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    pts = (dirs * 0.55).astype(np.float32)
    if backend == "jax":
        import jax.numpy as jnp
        proj = np.asarray(grid.project(jnp.asarray(pts), n_steps=4, backend="jax"))
    else:
        proj = np.asarray(grid.project(pts, n_steps=4, backend="numpy"))
    radii = np.linalg.norm(proj, axis=-1)
    # 5 mm tolerance across all directions on the 96-cube sphere.
    np.testing.assert_allclose(radii, 0.4, atol=5e-3)


# --- mesh bake (skipped if trimesh missing) --------------------------------


def test_from_mesh_sphere_against_analytic() -> None:
    trimesh = pytest.importorskip("trimesh")
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    grid = SDFGrid3D.from_mesh(sphere, spacing=0.05, padding=0.2)
    # Sample points on a sphere of radius 0.7 (outside) — true SDF = 0.2.
    rng = np.random.RandomState(0)
    dirs = rng.randn(32, 3); dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    pts = (dirs * 0.7).astype(np.float32)
    phi = grid.sdf(pts, backend="numpy")
    # Mesh→voxel discretization + icosphere approximation: ~5cm tolerance.
    np.testing.assert_allclose(phi, 0.2, atol=5e-2)


def test_contains_distance_gradient() -> None:
    grid = _sphere_grid(N=64)
    p_in = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    p_out = np.array([0.6, 0.0, 0.0], dtype=np.float32)
    assert grid.contains(p_in)
    assert not grid.contains(p_out)
    assert grid.distance(p_out) > 0
    g = grid.gradient(p_out)
    assert g.shape == (3,)
    np.testing.assert_allclose(np.linalg.norm(g), 1.0, atol=2e-2)
