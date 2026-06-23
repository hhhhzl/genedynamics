"""Parametric contact surfaces for surface-contact manipulation.

A surface is `p_s(ξ,η) ∈ R^3` with `(ξ,η)∈[0,1]^2` (idea.txt eq:exp_arm). The unit
normal `n_s = (∂_ξ p × ∂_η p)/‖·‖` and the tangents are obtained by autodiff of
`point` — so only the closed-form `point` is written per family; normals stay
exact and differentiable (clean-state Jacobians for the MDAC `geometry_fn`).

Families:
  plane, cylinder, ellipsoid  (analytic, here)
  NURBS bumps / unseen        (jax NURBS basis — TODO, later)

Upstream (`core/coverage`) — reusable by any manipulation env. Pure JAX.
The `Surface.kind` is a static python string (used as a dispatch key at trace
time); `params` are jnp arrays. Close a `Surface` over the env (static config).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Surface:
    """A parametric surface patch. `params` is a tuple of jnp arrays whose layout
    depends on `kind`. `extent` scales (ξ,η)∈[0,1]^2 to the physical patch."""
    kind: str          # "plane" | "cylinder" | "ellipsoid"
    params: Tuple      # kind-specific arrays


# --- closed-form points (each maps (ξ,η)∈[0,1]^2 -> R^3) --------------------

def _plane_point(params, xi, eta):
    origin, u_axis, v_axis = params           # (3,),(3,),(3,) (axes carry extent)
    return origin + xi * u_axis + eta * v_axis


def _cylinder_point(params, xi, eta):
    # params: center(3), radius(), ang0(), ang_span(), height(), axis_z(3 unit), x_dir(3 unit)
    center, radius, ang0, ang_span, height, axis_z, x_dir = params
    y_dir = jnp.cross(axis_z, x_dir)
    ang = ang0 + xi * ang_span
    radial = jnp.cos(ang) * x_dir + jnp.sin(ang) * y_dir
    return center + radius * radial + (eta - 0.5) * height * axis_z


def _ellipsoid_point(params, xi, eta):
    # params: center(3), radii(3); xi->azimuth patch, eta->polar patch (front cap)
    center, radii, az0, az_span, po0, po_span = params
    az = az0 + xi * az_span
    po = po0 + eta * po_span
    local = jnp.array([
        jnp.sin(po) * jnp.cos(az),
        jnp.sin(po) * jnp.sin(az),
        jnp.cos(po),
    ])
    return center + radii * local


_POINT = {"plane": _plane_point, "cylinder": _cylinder_point, "ellipsoid": _ellipsoid_point}


# --- differential geometry (autodiff of point) ------------------------------

def point(surf: Surface, xi, eta) -> jnp.ndarray:
    return _POINT[surf.kind](surf.params, jnp.asarray(xi), jnp.asarray(eta))


def tangents(surf: Surface, xi, eta) -> Tuple[jnp.ndarray, jnp.ndarray]:
    f = lambda a, b: point(surf, a, b)
    d_xi = jax.jacfwd(f, argnums=0)(jnp.asarray(xi, jnp.float32), jnp.asarray(eta, jnp.float32))
    d_eta = jax.jacfwd(f, argnums=1)(jnp.asarray(xi, jnp.float32), jnp.asarray(eta, jnp.float32))
    return d_xi, d_eta


def normal(surf: Surface, xi, eta) -> jnp.ndarray:
    """Unit surface normal `n_s` (eq:surface_normal). Outward by the (∂ξ×∂η)
    convention of each family's parametrization."""
    d_xi, d_eta = tangents(surf, xi, eta)
    n = jnp.cross(d_xi, d_eta)
    return n / (jnp.linalg.norm(n) + 1e-9)


# --- convenience factories ------------------------------------------------

def make_plane(origin=(0.45, -0.2, 0.4), u=(0.0, 0.4, 0.0), v=(0.0, 0.0, 0.3)) -> Surface:
    """Flat patch. `u,v` span the patch; normal ≈ u×v."""
    return Surface("plane", (jnp.asarray(origin, jnp.float32),
                             jnp.asarray(u, jnp.float32), jnp.asarray(v, jnp.float32)))


def make_cylinder(center=(0.5, 0.0, 0.4), radius=0.15, ang0=-0.6, ang_span=1.2,
                  height=0.3, axis_z=(0.0, 0.0, 1.0), x_dir=(1.0, 0.0, 0.0)) -> Surface:
    """Cylindrical patch."""
    return Surface("cylinder", (
        jnp.asarray(center, jnp.float32), jnp.asarray(radius, jnp.float32),
        jnp.asarray(ang0, jnp.float32), jnp.asarray(ang_span, jnp.float32),
        jnp.asarray(height, jnp.float32), jnp.asarray(axis_z, jnp.float32),
        jnp.asarray(x_dir, jnp.float32)))


def make_ellipsoid(center=(0.5, 0.0, 0.35), radii=(0.18, 0.18, 0.22),
                   az0=-0.6, az_span=1.2, po0=0.6, po_span=0.9) -> Surface:
    """Convex ellipsoid patch."""
    return Surface("ellipsoid", (
        jnp.asarray(center, jnp.float32), jnp.asarray(radii, jnp.float32),
        jnp.asarray(az0, jnp.float32), jnp.asarray(az_span, jnp.float32),
        jnp.asarray(po0, jnp.float32), jnp.asarray(po_span, jnp.float32)))


# named surface -> Surface; NURBS is a later addition
def surface_for_level(level: str) -> Surface:
    level = str(level).lower()
    if level in ("plane",):
        return make_plane()
    if level in ("cylinder",):
        return make_cylinder()
    if level in ("ellipsoid",):
        return make_ellipsoid()
    raise NotImplementedError(f"surface '{level}' not implemented (NURBS = TODO)")


__all__ = [
    "Surface", "point", "tangents", "normal",
    "make_plane", "make_cylinder", "make_ellipsoid", "surface_for_level",
]
