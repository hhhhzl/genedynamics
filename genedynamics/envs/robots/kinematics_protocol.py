"""Kinematics protocol — backend-agnostic interface for serial-arm kinematics.

Implementations:
  * `JAXKinematics` (genedynamics.envs.robots.jax_kinematics) — pure JAX, jit/vmap/grad.
  * `ManipulatorModel` (genedynamics.envs.robots.manipulator) — host-side numpy
    via PyBullet / Pinocchio. Conforms to this protocol via duck typing.

Future runtime backends (torch, mlx, …) implement the same protocol; downstream
planners and CBF terms consume `KinematicsProtocol` instead of any concrete class.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Tuple, runtime_checkable


# Backend-agnostic array alias. Concrete implementations may use np.ndarray,
# jnp.ndarray, torch.Tensor — downstream code does not branch on type.
ArrayLike = Any


@runtime_checkable
class KinematicsProtocol(Protocol):
    """Forward kinematics + geometric Jacobian for a serial robot arm.

    All return arrays are in the same backend as the input `q` (no host
    roundtrip). Conforming implementations must be safe under jit/vmap when
    they target a device backend.
    """

    n_dof: int

    def fk(self, q: ArrayLike) -> ArrayLike:
        """Forward kinematics.

        Args:
            q: joint positions, shape ``(n_dof,)``.

        Returns:
            ``(4, 4)`` homogeneous transform of the end-effector frame
            expressed in the base frame.
        """
        ...

    def jacobian(self, q: ArrayLike) -> ArrayLike:
        """Geometric Jacobian.

        Args:
            q: joint positions, shape ``(n_dof,)``.

        Returns:
            ``(6, n_dof)`` Jacobian. Rows 0–2 are linear-velocity contributions
            (``J_p``); rows 3–5 are angular-velocity contributions (``J_R``),
            both expressed in the base frame.
        """
        ...

    def jacobian_pinv(self, q: ArrayLike, damping: float = 0.0) -> ArrayLike:
        """Damped pseudo-inverse of the geometric Jacobian.

        ``J† = J^T (J J^T + λI)^{-1}`` with ``λ = damping**2``.

        Args:
            q: joint positions, shape ``(n_dof,)``.
            damping: Tikhonov damping; non-zero stabilises near singularities.

        Returns:
            ``(n_dof, 6)`` matrix.
        """
        ...

    def nullspace(self, q: ArrayLike) -> ArrayLike:
        """Joint-space nullspace projector.

        ``N(q) = I - J†(q) J(q)``; columns span the redundancy of ``q`` w.r.t.
        the end-effector twist.

        Returns:
            ``(n_dof, n_dof)`` projector.
        """
        ...

    @property
    def joint_limits(self) -> Tuple[ArrayLike, ArrayLike]:
        """``(lo, hi)`` joint position bounds, each shape ``(n_dof,)``."""
        ...

    @property
    def velocity_limits(self) -> Tuple[ArrayLike, ArrayLike]:
        """``(lo, hi)`` joint velocity bounds, each shape ``(n_dof,)``."""
        ...
