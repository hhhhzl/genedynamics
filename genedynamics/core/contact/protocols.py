"""Generic contact protocols for end-effector / surface contact tasks.

The contact subsystem exposes three pieces:

  * :class:`SdfQuery` — the minimum surface representation a contact model
    needs. Lives here as a duck-type Protocol so this package does **not**
    import :mod:`genedynamics.envs.obstacles`; concrete SDFs (e.g.
    ``SDFGrid3D``) satisfy the protocol structurally.
  * :class:`ContactState` — a frozen, backend-agnostic snapshot of a single
    contact query.
  * :class:`ContactModel` — the interface concrete contact models implement.

Designed to be runtime-backend-agnostic: ``ArrayLike`` is whatever the caller
passed in (numpy / jax / torch arrays). Concrete implementations dispatch
inside via the same ``Backend = Literal[...]`` pattern used elsewhere in the
package (see ``envs.obstacles.sdf_grid_3d`` for the canonical example).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Tuple, runtime_checkable


# Backend-agnostic alias — concrete implementations may use np.ndarray,
# jnp.ndarray, torch.Tensor. Downstream code does not branch on type.
ArrayLike = Any


@runtime_checkable
class SdfQuery(Protocol):
    """Minimum SDF interface a contact model needs.

    Any object exposing :meth:`sdf_and_grad` with the signature below
    structurally satisfies this protocol — the contact module never imports
    a concrete SDF class. ``SDFGrid3D`` and the 2D ``SDFTexture2D`` both
    satisfy it.
    """

    def sdf_and_grad(self, points: ArrayLike, *, backend: str = "numpy") -> Tuple[ArrayLike, ArrayLike]:
        """Return ``(phi, grad_phi)`` at ``points``.

        ``phi`` shape ``(...,)`` and ``grad_phi`` shape ``(..., 3)``; sign
        convention is **positive outside, negative inside** (so penetration
        depth is ``-phi``).
        """
        ...


@dataclass(frozen=True)
class ContactState:
    """Snapshot of a single contact query.

    All fields are arrays in the caller's runtime backend; shapes are given
    for a single query — implementations should batch by stacking.

    Attributes:
        in_contact: shape ``()``, bool — whether the tool is in/penetrating
            the surface (``phi <= contact_threshold``).
        surface_point: ``(3,)`` projection :math:`\\Pi_M(p_{\\rm tool})` onto
            the zero level set; equals ``p_tool - phi * n``.
        normal: ``(3,)`` outward unit normal at the surface point;
            ``n = ∇φ / ‖∇φ‖``.
        tangent_projector: ``(3, 3)`` tangent plane projector
            ``P_T = I - n n^T``; ``P_T v`` gives the surface-tangential part of
            ``v``.
        penetration: ``()`` scalar penetration depth ``d = max(-phi, 0)``;
            non-negative.
        penetration_rate: ``()`` scalar normal compression rate
            ``\\dot d = -∇φ · v_{\\rm tool}``; positive when pressing in.
        force: ``(3,)`` contact force on the tool in world frame (zero outside
            contact). Sign convention: force points **away** from the surface
            (i.e. surface pushes tool out).
    """

    in_contact: ArrayLike
    surface_point: ArrayLike
    normal: ArrayLike
    tangent_projector: ArrayLike
    penetration: ArrayLike
    penetration_rate: ArrayLike
    force: ArrayLike


@runtime_checkable
class ContactModel(Protocol):
    """Generic end-effector / surface contact model.

    Implementations decide the force law (compliant spring-damper, rigid with
    Coulomb friction, etc.). All take a tool world-frame position and
    velocity plus an :class:`SdfQuery` describing the environment, and return
    a :class:`ContactState` snapshot.

    The protocol does **not** prescribe a backend; concrete implementations
    expose a ``backend`` keyword (``"numpy"``/``"jax"``/``"torch"``) on
    construction, and ``query`` outputs arrays in that backend.
    """

    def query(
        self,
        tool_position: ArrayLike,    # (3,) or (..., 3)
        tool_velocity: ArrayLike,    # (3,) or (..., 3)
        env_sdf: SdfQuery,
    ) -> ContactState: ...
