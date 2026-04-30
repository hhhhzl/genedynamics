"""Rigid contact with Coulomb friction.

Models a rigid contact where the normal force is whatever the constraint
solver decides (so this model itself does **not** produce a normal force —
it produces the kinematic state and a friction-cone bound). The tangential
force satisfies ``‖f_t‖ ≤ μ f_n`` (linearised pyramid) and is reported as
zero by default; downstream solvers (e.g. WBC, force-CBF QP) compose the
returned :class:`ContactState` with their own dynamics to close the loop.

Use this when you want the contact-state plumbing (Π_M, n, P_T, penetration)
without committing to a compliant force law — e.g. for stiff manipulation
where contact force is determined by the QP, not a spring.
"""

from __future__ import annotations

from typing import Any, Literal

from genedynamics.core.contact.protocols import ContactModel, ContactState, SdfQuery
from genedynamics.core.contact.spring_damper import (
    _astype,
    _broadcast_eye,
    _dot,
    _outer,
    _xp,
    Backend,
)


class RigidWithFrictionContact(ContactModel):
    """Rigid contact with a friction-cone bound (no compliant force law).

    Args:
        mu: Coulomb friction coefficient ``μ ≥ 0``. Stored as :attr:`mu` for
            downstream solvers that compose ``‖f_t‖ ≤ μ f_n``.
        contact_threshold: ``phi`` threshold below which contact is active.
        backend: ``"numpy"``/``"jax"``/``"torch"``.

    Notes:
        :meth:`query` returns ``force = 0`` and the kinematic state. Solvers
        that need an actual contact force should add a normal-force decision
        variable and impose ``‖f_t‖ ≤ μ f_n`` using :attr:`mu` and
        :attr:`ContactState.tangent_projector`.
    """

    def __init__(
        self,
        mu: float,
        *,
        contact_threshold: float = 0.0,
        backend: Backend = "numpy",
    ) -> None:
        self.mu = float(mu)
        self.contact_threshold = float(contact_threshold)
        self.backend = backend
        self._xp = _xp(backend)

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
        grad_norm = xp.linalg.norm(grad, axis=-1, keepdims=True) if hasattr(xp, "linalg") \
            else grad.norm(dim=-1, keepdim=True)
        eps = xp.asarray(1e-8, dtype=p.dtype)
        n = grad / xp.maximum(grad_norm, eps)
        phi_b = phi[..., None] if phi.ndim < p.ndim else phi
        surface_point = p - phi_b * n

        nn = _outer(xp, n, n)
        eye3 = xp.eye(3, dtype=p.dtype)
        if nn.ndim > 2:
            eye3 = _broadcast_eye(xp, nn.shape[:-2], 3, dtype=p.dtype)
        tangent_projector = eye3 - nn

        in_contact = phi <= self.contact_threshold
        zero = xp.asarray(0.0, dtype=p.dtype)
        penetration = xp.maximum(self.contact_threshold - phi, zero)
        penetration_rate = -_dot(xp, grad, v)

        # Rigid model leaves force determination to the downstream solver.
        force = xp.zeros_like(p)

        return ContactState(
            in_contact=in_contact,
            surface_point=surface_point,
            normal=n,
            tangent_projector=tangent_projector,
            penetration=penetration,
            penetration_rate=penetration_rate,
            force=force,
        )
