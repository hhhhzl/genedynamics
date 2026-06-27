"""Parametric contact surfaces for surface-contact manipulation.

A surface is `p_s(ξ,η) ∈ R^3` with `(ξ,η)∈[0,1]^2` (idea.txt eq:exp_arm). The unit
normal `n_s = (∂_ξ p × ∂_η p)/‖·‖` and the tangents are obtained by autodiff of
`point` — so only the closed-form `point` is written per family; normals stay
exact and differentiable (clean-state Jacobians for the MDAC `geometry_fn`).

Families:
  plane, cylinder, ellipsoid  (analytic)
  nurbs                       (5x5 tensor-product cubic NURBS height field; the
                               convex / bumpy / unseen surface families S2-S4)

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


# --- NURBS height-field patch (S2 convex / S3 bumpy / S4 unseen) ------------
# Tensor-product cubic NURBS over an n x n control grid of HEIGHTS along the
# patch base normal. The base (x,y) is the flat patch `origin + xi*u + eta*v`;
# the control heights H_ij (+ optional rational weights W_ij) shape the surface.
# `point` stays closed-form, so `tangents`/`normal` (autodiff of `point`) give
# the exact analytic normal field for free — no per-family normal code.

_NURBS_DEGREE = 3
_NURBS_N = 5


def _clamped_knots(n_ctrl: int, degree: int) -> jnp.ndarray:
    """Clamped uniform knot vector, length n_ctrl + degree + 1."""
    n_internal = n_ctrl - degree - 1
    internal = jnp.linspace(0.0, 1.0, n_internal + 2)[1:-1] if n_internal > 0 else jnp.zeros((0,))
    return jnp.concatenate([jnp.zeros((degree + 1,)), internal, jnp.ones((degree + 1,))]).astype(jnp.float32)


def _bspline_basis(t, knots: jnp.ndarray, degree: int) -> jnp.ndarray:
    """Cox-de Boor basis N_i(t), i=0..n_ctrl-1, at scalar t (autodiff-safe:
    division guarded; degree-0 indicator has measure-zero gradient, so jacfwd
    yields the exact piecewise-polynomial derivative a.e.)."""
    t = jnp.asarray(t, jnp.float32)
    nspan = knots.shape[0] - 1
    left, right = knots[:-1], knots[1:]
    N = jnp.where((t >= left) & (t < right), 1.0, 0.0)                 # degree 0
    # clamped right end: t == last knot belongs to the last NON-degenerate span
    last_span = (right >= knots[-1]) & (left < right)
    N = N + jnp.where(t >= knots[-1], last_span.astype(jnp.float32), 0.0)
    for d in range(1, degree + 1):
        cnt = nspan - d
        i = jnp.arange(cnt)
        d1 = knots[i + d] - knots[i]
        d2 = knots[i + d + 1] - knots[i + 1]
        a = jnp.where(d1 > 1e-9, (t - knots[i]) / jnp.where(d1 > 1e-9, d1, 1.0), 0.0)
        b = jnp.where(d2 > 1e-9, (knots[i + d + 1] - t) / jnp.where(d2 > 1e-9, d2, 1.0), 0.0)
        N = a * N[:cnt] + b * N[1:cnt + 1]
    return N                                                          # (n_ctrl,)


def _nurbs_height(xi, eta, ctrl_h, weights, knots) -> jnp.ndarray:
    """Rational tensor-product height z(xi,eta) over the control grid."""
    nu = _bspline_basis(xi, knots, _NURBS_DEGREE)                     # (n,)
    nv = _bspline_basis(eta, knots, _NURBS_DEGREE)                    # (n,)
    b = (nu[:, None] * nv[None, :]) * weights                         # (n,n)
    return jnp.sum(b * ctrl_h) / (jnp.sum(b) + 1e-9)


def _nurbs_point(params, xi, eta):
    # params: origin(3), u(3), v(3), ctrl_h(n,n), weights(n,n), knots(n+4,)
    origin, u_axis, v_axis, ctrl_h, weights, knots = params
    base = origin + xi * u_axis + eta * v_axis
    n_base = jnp.cross(u_axis, v_axis)
    n_base = n_base / (jnp.linalg.norm(n_base) + 1e-9)
    return base + _nurbs_height(xi, eta, ctrl_h, weights, knots) * n_base


_POINT = {"plane": _plane_point, "cylinder": _cylinder_point,
          "ellipsoid": _ellipsoid_point, "nurbs": _nurbs_point}


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

def make_plane(origin=(0.3, -0.15, 0.65), u=(0.3, 0.0, 0.0), v=(0.0, 0.3, 0.0)) -> Surface:
    """Flat HORIZONTAL patch (u,v span x,y -> normal ≈ +z), scanned from above so
    a down-pointing tool aligns. `u,v` span the patch; normal ≈ u×v."""
    return Surface("plane", (jnp.asarray(origin, jnp.float32),
                             jnp.asarray(u, jnp.float32), jnp.asarray(v, jnp.float32)))


def make_cylinder(center=(0.45, 0.0, 0.5), radius=0.15, ang0=-0.6, ang_span=1.2,
                  height=0.3, axis_z=(0.0, 1.0, 0.0), x_dir=(0.0, 0.0, 1.0)) -> Surface:
    """Horizontal cylinder (axis along y) scanned over the TOP arc -> normal ≈ +z."""
    return Surface("cylinder", (
        jnp.asarray(center, jnp.float32), jnp.asarray(radius, jnp.float32),
        jnp.asarray(ang0, jnp.float32), jnp.asarray(ang_span, jnp.float32),
        jnp.asarray(height, jnp.float32), jnp.asarray(axis_z, jnp.float32),
        jnp.asarray(x_dir, jnp.float32)))


def make_ellipsoid(center=(0.45, 0.0, 0.5), radii=(0.18, 0.18, 0.22),
                   az0=-0.6, az_span=1.2, po0=0.1, po_span=0.6) -> Surface:
    """Convex ellipsoid TOP cap (polar near the +z pole) -> normal ≈ +z."""
    return Surface("ellipsoid", (
        jnp.asarray(center, jnp.float32), jnp.asarray(radii, jnp.float32),
        jnp.asarray(az0, jnp.float32), jnp.asarray(az_span, jnp.float32),
        jnp.asarray(po0, jnp.float32), jnp.asarray(po_span, jnp.float32)))


# --- NURBS surface families (5x5 control grid), seed -> deterministic patch --

# HORIZONTAL patch (u,v span x,y -> base normal +z): a bumpy "tabletop" scanned
# from above, so a down-pointing tool aligns. Height field rises along +z.
_NURBS_ORIGIN = (0.3, -0.2, 0.6)       # corner near the robot workspace
_NURBS_U = (0.4, 0.0, 0.0)             # 0.4 m span (x)
_NURBS_V = (0.0, 0.4, 0.0)             # 0.4 m span (y) -> 0.4 x 0.4 m patch


def _grid2d(n: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    g = jnp.linspace(0.0, 1.0, n)
    return jnp.meshgrid(g, g, indexing="ij")


def _convex_heights(amp: float, n: int = _NURBS_N) -> jnp.ndarray:
    """Smooth convex cap (paraboloid bulge along the base normal), peak `amp`."""
    uu, vv = _grid2d(n)
    return jnp.clip(amp * (1.0 - 4.0 * ((uu - 0.5) ** 2 + (vv - 0.5) ** 2)), 0.0, amp)


def _make_nurbs(ctrl_h: jnp.ndarray) -> Surface:
    n = ctrl_h.shape[0]
    return Surface("nurbs", (
        jnp.asarray(_NURBS_ORIGIN, jnp.float32), jnp.asarray(_NURBS_U, jnp.float32),
        jnp.asarray(_NURBS_V, jnp.float32), ctrl_h.astype(jnp.float32),
        jnp.ones((n, n), jnp.float32), _clamped_knots(n, _NURBS_DEGREE)))


def make_nurbs_convex(seed: int = 0, amp_lo: float = 0.08, amp_hi: float = 0.12) -> Surface:
    """`convex`: smooth convex NURBS (abdomen/ellipsoid-like), peak height in [lo,hi]."""
    amp = amp_lo + (amp_hi - amp_lo) * jax.random.uniform(jax.random.PRNGKey(seed))
    return _make_nurbs(_convex_heights(amp))


def make_nurbs_bumpy(seed: int = 0, base_amp: float = 0.08, bump_amp: float = 0.04,
                     n_bumps: int = 3, sigma: float = 0.15) -> Surface:
    """`bumpy`: convex base + random Gaussian bumps/valleys (nonconvex)."""
    uu, vv = _grid2d(_NURBS_N)
    h = _convex_heights(base_amp)
    k = jax.random.PRNGKey(seed)
    for j in range(n_bumps):
        k, kc, ka = jax.random.split(k, 3)
        c = jax.random.uniform(kc, (2,))
        a = jax.random.uniform(ka, (), minval=-bump_amp, maxval=bump_amp)
        h = h + a * jnp.exp(-((uu - c[0]) ** 2 + (vv - c[1]) ** 2) / (2.0 * sigma ** 2))
    return _make_nurbs(h)


def make_nurbs_unseen(seed: int = 0, amp: float = 0.06) -> Surface:
    """`unseen`: held-out random NURBS geometry, control heights in [-amp,amp]
    (env adds surface-stiffness / friction DR)."""
    h = jax.random.uniform(jax.random.PRNGKey(seed), (_NURBS_N, _NURBS_N),
                           minval=-amp, maxval=amp)
    return _make_nurbs(h)


def translate(surf: Surface, shift) -> Surface:
    """Translate a surface by ``shift`` (R^3). Every family stores its base point
    (plane origin / cylinder & ellipsoid center / NURBS origin) as ``params[0]``,
    so shifting that point rigidly translates the whole patch."""
    shift = jnp.asarray(shift, jnp.float32)
    p = list(surf.params)
    p[0] = jnp.asarray(p[0], jnp.float32) + shift
    return Surface(surf.kind, tuple(p))


# named surface family -> Surface. Descriptive names only (no S1-S4 codes):
#   plane / cylinder  (analytic),  convex / bumpy / unseen  (NURBS).
def surface_for_level(level: str, seed: int = 0) -> Surface:
    level = str(level).lower()
    if level == "plane":
        return make_plane()
    if level == "cylinder":
        return make_cylinder()
    if level == "ellipsoid":                          # analytic, kept for back-compat
        return make_ellipsoid()
    if level == "convex":
        return make_nurbs_convex(seed)
    if level == "bumpy":
        return make_nurbs_bumpy(seed)
    if level == "unseen":
        return make_nurbs_unseen(seed)
    raise NotImplementedError(f"surface family '{level}' not recognized")


__all__ = [
    "Surface", "point", "tangents", "normal",
    "make_plane", "make_cylinder", "make_ellipsoid",
    "make_nurbs_convex", "make_nurbs_bumpy", "make_nurbs_unseen", "translate", "surface_for_level",
]
