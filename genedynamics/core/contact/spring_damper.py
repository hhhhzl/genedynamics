"""Compliant spring-damper contact model.

Force law (along surface normal, no tangential friction):

.. math::
    f = (k \\cdot d + b \\cdot \\dot d) \\cdot n,
    \\quad d = \\max(-\\phi, 0), \\quad \\dot d = -\\nabla \\phi \\cdot v_{\\rm tool}

Outside contact (``phi > 0``) the force is zero. The stiffness ``k`` may be
constant or a callable evaluated at the surface point — useful for spatially
varying compliance (e.g. soft tissue with site-dependent stiffness in
ultrasound scanning).

Backend dispatch (``numpy``/``jax``/``torch``) selected at construction so
``query`` is a single jit-friendly call.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Union

import numpy as np

try:
    import jax.numpy as jnp
except Exception:  # pragma: no cover
    jnp = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from genedynamics.core.contact.protocols import ContactModel, ContactState, SdfQuery


Backend = Literal["numpy", "jax", "torch"]


_BACKEND_MOD = {"numpy": np}
if jnp is not None:
    _BACKEND_MOD["jax"] = jnp
if torch is not None:
    _BACKEND_MOD["torch"] = torch


def _xp(backend: Backend):
    """Resolve the array module for a backend identifier."""
    mod = _BACKEND_MOD.get(backend)
    if mod is None:
        raise RuntimeError(
            f"backend {backend!r} not available; install jax or torch to enable it."
        )
    return mod


class SpringDamperContact(ContactModel):
    """Spring-damper compliant contact model.

    Args:
        stiffness: spring constant ``k`` (scalar) or callable
            ``k(surface_point) -> scalar`` for spatially varying stiffness.
            Units: force per metre.
        damping: damping coefficient ``b`` (scalar). Units: force per (m/s).
        contact_threshold: ``phi`` threshold below which we count as in
            contact. Defaults to 0; set positive to model "near-contact"
            (e.g. ultrasound coupling) or negative for hysteresis.
        backend: ``"numpy"`` (default), ``"jax"``, or ``"torch"``.
    """

    def __init__(
        self,
        stiffness: Union[float, Callable[[Any], Any]],
        damping: float,
        *,
        contact_threshold: float = 0.0,
        backend: Backend = "numpy",
    ) -> None:
        self.stiffness = stiffness
        self.damping = float(damping)
        self.contact_threshold = float(contact_threshold)
        self.backend = backend
        self._xp = _xp(backend)

    def _stiffness_at(self, surface_point: Any) -> Any:
        if callable(self.stiffness):
            return self.stiffness(surface_point)
        return self._xp.asarray(self.stiffness, dtype=surface_point.dtype)

    def query(
        self,
        tool_position: Any,
        tool_velocity: Any,
        env_sdf: SdfQuery,
    ) -> ContactState:
        xp = self._xp
        p = xp.asarray(tool_position)
        v = xp.asarray(tool_velocity)

        phi, grad = env_sdf.sdf_and_grad(p, backend=self.backend)
        # Robust unit normal — guard against the ‖∇φ‖→0 case at degenerate
        # points (well inside thick obstacles, far from any surface).
        grad_norm = xp.linalg.norm(grad, axis=-1, keepdims=True) if hasattr(xp, "linalg") \
            else _norm_torch(grad)
        eps = xp.asarray(1e-8, dtype=p.dtype)
        n = grad / xp.maximum(grad_norm, eps)
        # Surface point: p_s = p - phi * n.
        phi_b = phi[..., None] if phi.ndim < p.ndim else phi
        surface_point = p - phi_b * n

        # P_T = I - n n^T (3x3 in the trailing dim).
        nn = _outer(xp, n, n)
        eye3 = xp.eye(3, dtype=p.dtype)
        # Broadcast eye3 to leading shape.
        if nn.ndim > 2:
            eye3 = _broadcast_eye(xp, nn.shape[:-2], 3, dtype=p.dtype)
        tangent_projector = eye3 - nn

        in_contact = phi <= self.contact_threshold
        # Penetration depth (≥ 0 inside contact, 0 outside).
        zero = xp.asarray(0.0, dtype=p.dtype)
        penetration = xp.maximum(self.contact_threshold - phi, zero)
        # Compression rate ḋ = -∇φ · v (positive when pressing in further).
        penetration_rate = -_dot(xp, grad, v)

        # Force law along normal; gated to zero outside contact via in_contact.
        k = self._stiffness_at(surface_point)
        if k.ndim < penetration.ndim:
            k = _broadcast_to(xp, k, penetration.shape, p.dtype)
        b = xp.asarray(self.damping, dtype=p.dtype)
        f_mag = k * penetration + b * penetration_rate
        gate = _astype(xp, in_contact, p.dtype)
        f_mag = f_mag * gate
        force = f_mag[..., None] * n

        return ContactState(
            in_contact=in_contact,
            surface_point=surface_point,
            normal=n,
            tangent_projector=tangent_projector,
            penetration=penetration,
            penetration_rate=penetration_rate,
            force=force,
        )


# ----- small backend-agnostic helpers (avoid hard-coding numpy/jnp/torch) ----


def _outer(xp, a: Any, b: Any) -> Any:
    """Outer product on the last axis: ``a[..., :, None] * b[..., None, :]``."""
    return a[..., :, None] * b[..., None, :]


def _dot(xp, a: Any, b: Any) -> Any:
    """Inner product on the last axis."""
    return (a * b).sum(axis=-1)


def _norm_torch(grad: Any) -> Any:
    return grad.norm(dim=-1, keepdim=True)


def _broadcast_eye(xp, leading_shape, n: int, dtype: Any) -> Any:
    eye_n = xp.eye(n, dtype=dtype)
    target = tuple(leading_shape) + (n, n)
    if hasattr(xp, "broadcast_to"):
        return xp.broadcast_to(eye_n, target)
    # torch fallback
    return eye_n.expand(target)


def _broadcast_to(xp, a: Any, shape, dtype: Any) -> Any:
    a = xp.asarray(a, dtype=dtype)
    if hasattr(xp, "broadcast_to"):
        return xp.broadcast_to(a, shape)
    return a.expand(shape)


def _astype(xp, a: Any, dtype: Any) -> Any:
    if hasattr(a, "astype"):
        return a.astype(dtype)
    return a.to(dtype=dtype)
