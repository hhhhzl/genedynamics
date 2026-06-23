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
    for (xi, eta) in [(0.0, 0.5), (0.5, 0.5), (1.0, 0.5)]:
        p = sg.point(surf, xi, eta)
        radial = p - center
        radial = radial.at[2].set(0.0)               # drop axis component
        assert abs(float(jnp.linalg.norm(radial)) - 0.15) < 1e-5


def test_differentiable_through_normal():
    surf = sg.make_ellipsoid()
    g = jax.grad(lambda xi: jnp.sum(sg.normal(surf, xi, 0.4)))(jnp.asarray(0.5))
    assert jnp.all(jnp.isfinite(g))


def test_level_dispatch():
    assert sg.surface_for_level("plane").kind == "plane"
    assert sg.surface_for_level("ellipsoid").kind == "ellipsoid"
    with pytest.raises(NotImplementedError):
        sg.surface_for_level("nurbs")
