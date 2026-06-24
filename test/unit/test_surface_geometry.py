"""Unit gates for core/coverage/surface_geometry (plane/cylinder/ellipsoid). fedguide CPU.

NOTE: deliberately does NOT enable jax x64 globally — that pollutes the whole
pytest session (jax config is process-global) and breaks float32 byte-identity
gates in other test files. float32 tolerances (1e-5) are sufficient here."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from genedynamics.core.coverage import surface_geometry as sg


@pytest.mark.parametrize("surf,expect_n", [
    (sg.make_plane(origin=(0, 0, 0), u=(0, 1, 0), v=(0, 0, 1)), (1.0, 0.0, 0.0)),
])
def test_plane_point_and_normal(surf, expect_n):
    # plane point is affine in (xi,eta)
    assert jnp.allclose(sg.point(surf, 0.0, 0.0), jnp.zeros(3))
    assert jnp.allclose(sg.point(surf, 1.0, 0.0), jnp.array([0, 1.0, 0]))
    n = sg.normal(surf, 0.3, 0.7)
    # normal is u x v = (0,1,0)x(0,0,1) = (1,0,0); unit
    assert jnp.allclose(jnp.abs(n), jnp.abs(jnp.asarray(expect_n)), atol=1e-5)
    assert abs(float(jnp.linalg.norm(n)) - 1.0) < 1e-5


def test_normal_is_unit_and_orthogonal_to_tangents():
    for surf in (sg.make_plane(), sg.make_cylinder(), sg.make_ellipsoid()):
        for (xi, eta) in [(0.2, 0.3), (0.7, 0.5), (0.9, 0.1)]:
            d_xi, d_eta = sg.tangents(surf, xi, eta)
            n = sg.normal(surf, xi, eta)
            assert abs(float(jnp.linalg.norm(n)) - 1.0) < 1e-5
            assert abs(float(n @ d_xi)) < 1e-5          # normal ⟂ tangents
            assert abs(float(n @ d_eta)) < 1e-5


def test_cylinder_points_lie_on_radius():
    surf = sg.make_cylinder(center=(0.5, 0.0, 0.4), radius=0.15)
    center = jnp.array([0.5, 0.0, 0.4])
    axis = surf.params[5]                            # axis_z (now along y)
    for (xi, eta) in [(0.0, 0.5), (0.5, 0.5), (1.0, 0.5)]:
        p = sg.point(surf, xi, eta)
        radial = (p - center)
        radial = radial - jnp.dot(radial, axis) * axis   # drop the axis component
        assert abs(float(jnp.linalg.norm(radial)) - 0.15) < 1e-5


def test_differentiable_through_normal():
    surf = sg.make_ellipsoid()
    g = jax.grad(lambda xi: jnp.sum(sg.normal(surf, xi, 0.4)))(jnp.asarray(0.5))
    assert jnp.all(jnp.isfinite(g))


def test_level_dispatch():
    assert sg.surface_for_level("plane").kind == "plane"
    assert sg.surface_for_level("ellipsoid").kind == "ellipsoid"
    # S2/S3/S4 families -> NURBS height fields
    for fam in ("convex", "bumpy", "unseen"):
        assert sg.surface_for_level(fam, seed=0).kind == "nurbs"
    with pytest.raises(NotImplementedError):
        sg.surface_for_level("not_a_family")


def test_bspline_partition_of_unity():
    knots = sg._clamped_knots(sg._NURBS_N, sg._NURBS_DEGREE)
    for t in (0.0, 0.13, 0.5, 0.77, 1.0):
        N = sg._bspline_basis(jnp.asarray(t, jnp.float32), knots, sg._NURBS_DEGREE)
        assert abs(float(jnp.sum(N)) - 1.0) < 1e-5     # incl. clamped endpoints


def test_nurbs_normal_unit_orthogonal_and_matches_fd():
    def fd_normal(surf, xi, eta, h=1e-4):
        p = lambda a, b: np.asarray(sg.point(surf, a, b))
        d_xi = (p(xi + h, eta) - p(xi - h, eta)) / (2 * h)
        d_eta = (p(xi, eta + h) - p(xi, eta - h)) / (2 * h)
        n = np.cross(d_xi, d_eta)
        return n / (np.linalg.norm(n) + 1e-12)
    for fam in ("convex", "bumpy", "unseen"):
        surf = sg.surface_for_level(fam, seed=1)
        for (xi, eta) in [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]:
            d_xi, d_eta = sg.tangents(surf, xi, eta)
            n = sg.normal(surf, xi, eta)
            assert abs(float(jnp.linalg.norm(n)) - 1.0) < 1e-5
            assert abs(float(n @ d_xi)) < 1e-4 and abs(float(n @ d_eta)) < 1e-4
            assert np.max(np.abs(np.asarray(n) - fd_normal(surf, xi, eta))) < 2e-3


def test_nurbs_jit_grad_through_point():
    surf = sg.surface_for_level("bumpy", seed=2)
    g = jax.jit(jax.grad(lambda xi: sg.point(surf, xi, 0.35)[0]))(jnp.asarray(0.3, jnp.float32))
    assert jnp.isfinite(g)
