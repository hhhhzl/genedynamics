"""Pure-numpy core for the reference governor.

No I/O, no message types, no robot/task coupling: just the planar-frame
transforms and the proximal-projection math used by every governor backend.
Everything here is deterministic and unit-testable in isolation.

Convention: a planar reference is the 3-vector ``[s, n, psi]`` expressed in a
task frame (for a corridor: s = longitudinal, n = lateral, psi = yaw relative
to the frame). World references are ``[x, y, yaw]``.
"""

from __future__ import annotations

import numpy as np


def wrap_angle(a: float) -> float:
    """Wrap an angle (rad) to (-pi, pi]."""
    return (float(a) + np.pi) % (2.0 * np.pi) - np.pi


def to_frame(p_xy, yaw, origin, yaw_c) -> np.ndarray:
    """World (x, y, yaw) -> frame (s, n, psi) for a frame at ``origin`` / ``yaw_c``."""
    c, s = np.cos(-yaw_c), np.sin(-yaw_c)
    d = np.asarray(p_xy, dtype=float) - np.asarray(origin, dtype=float)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], wrap_angle(yaw - yaw_c)])


def from_frame(r_f, origin, yaw_c):
    """Frame (s, n, psi) -> world (xy, yaw)."""
    c, s = np.cos(yaw_c), np.sin(yaw_c)
    s_, n_ = float(r_f[0]), float(r_f[1])
    xy = np.asarray(origin, dtype=float) + np.array([c * s_ - s * n_, s * s_ + c * n_])
    return xy, wrap_angle(r_f[2] + yaw_c)


def proximal_delta(d_ref, q, eta) -> np.ndarray:
    """Unconstrained minimiser of ``q (d - d_ref)^2 + eta d^2`` per axis.

    Works in *delta space* (delta = r - r_prev), so eta pulls toward 0 (= stay
    at the previous governed reference). Returns ``q * d_ref / (q + eta)``.
    """
    q = np.asarray(q, dtype=float)
    return q * np.asarray(d_ref, dtype=float) / (q + float(eta))


def intersect_boxes(los, his):
    """Tightest box contained in all input boxes (lists of (3,) lo/hi vectors)."""
    lo = np.maximum.reduce([np.asarray(x, dtype=float) for x in los])
    hi = np.minimum.reduce([np.asarray(x, dtype=float) for x in his])
    return lo, hi


def clip(v, lo, hi) -> np.ndarray:
    return np.minimum(np.maximum(np.asarray(v, dtype=float), lo), hi)


def throttle_factor(err, deadband, gain) -> float:
    """Closed-loop rate throttle: 1.0 when on-track, shrinking as error grows."""
    over = max(0.0, float(err) - float(deadband))
    return float(np.clip(1.0 - float(gain) * over, 0.0, 1.0))


def project_box_halfspace(d, lo, hi, a, b, iters: int = 12):
    """Project ``d`` onto ``{box(lo, hi)} ∩ {a·d >= b}`` (delta space).

    ``a`` (3,) and ``b`` (scalar) define a single linear safety half-space on
    the reference delta. We use alternating projection (Dykstra-free, since the
    feasible set is convex and we only need a feasible-and-close point, not the
    exact Euclidean projection): clip to the box, then push along ``+a`` to
    satisfy the half-space, repeat. With a single half-space this converges in a
    handful of iterations; if the active axis is box-saturated it returns the
    best (box-feasible) point that maximises ``a·d`` — i.e. safety is favoured.

    Returns ``(d_proj, halfspace_active)`` where ``halfspace_active`` is True
    when the unconstrained box clip already violated ``a·d >= b`` (so the
    half-space actually moved the solution).
    """
    a = np.asarray(a, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    b = float(b)
    aa = float(np.dot(a, a))
    d_box = clip(d, lo, hi)
    if aa <= 1e-12:
        return d_box, False
    active = bool(np.dot(a, d_box) < b - 1e-12)
    if not active:
        return d_box, False
    d_cur = d_box
    for _ in range(int(iters)):
        slack = b - float(np.dot(a, d_cur))
        if slack <= 1e-9:
            break
        d_cur = d_cur + (slack / aa) * a  # project onto the half-space boundary
        d_cur = clip(d_cur, lo, hi)       # re-clip into the box
    return d_cur, True
