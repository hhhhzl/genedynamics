"""CBF helper rows for contact-bound constraints.

Produces linear inequality rows ``A u <= b`` that downstream QP filters
(e.g. ``core/constraints/convexify/cbf/CBFConvexifier``) can stack into
their existing convex constraint pipeline. We keep this **as helper
functions** rather than a bespoke ``Convexifier`` subclass so any planner
that already builds CBF rows for obstacles can append contact rows in the
same coordinate system without inheritance gymnastics.

Two helpers cover the common cases:

  * :func:`force_bound_rows` — bound the *normal* contact force to
    ``[f_min, f_max]`` for a spring-damper model. With
    ``f_n = k d + b ḋ`` and ``ḋ = -∇φ · v_tool``, taking the linearisation
    ``v_tool ≈ J(q) u`` gives a pair of linear rows in ``u``.

  * :func:`penetration_bound_rows` — bound the penetration depth
    ``d ∈ [d_min, d_max]`` via a CBF on ``φ``: ``φ̇ ≥ -α (φ - φ_target)``
    with ``φ̇ = ∇φ · v_tool = ∇φ · J u``.

Both return ``(A, b)`` shaped ``(2, n_dof)`` and ``(2,)`` (one row per
direction of the bound), in plain numpy by default. Pass arrays from any
runtime backend; the routine returns arrays in the same backend.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from genedynamics.core.contact.protocols import ContactState


def force_bound_rows(
    state: ContactState,
    jacobian: Any,
    *,
    stiffness: float,
    damping: float,
    f_min: float,
    f_max: float,
) -> Tuple[Any, Any]:
    """CBF rows for ``f_min ≤ f_n ≤ f_max`` under a spring-damper normal law.

    Setup. With ``d = max(-φ, 0)`` and ``ḋ = -∇φ · v_tool``,

        f_n = k d + b ḋ = k d - b (∇φ · J u).

    The bound ``f_n ≤ f_max`` becomes

        -b (∇φ · J) u ≤ f_max - k d
        ⇔  (b ∇φ J) u ≥ -(f_max - k d) - 0      (after sign flip)

    We return the canonical ``A u ≤ b`` form with two rows: one for the
    upper bound, one for the lower bound.

    Args:
        state: ``ContactState`` evaluated at the current ``q``.
        jacobian: ``(6, n_dof)`` geometric Jacobian or ``(3, n_dof)``
            position Jacobian. Top three rows are used.
        stiffness, damping: spring-damper coefficients matching the
            :class:`SpringDamperContact` that produced ``state``.
        f_min, f_max: normal-force bounds (``f_min`` may be 0 to enforce
            "do not pull on the surface").

    Returns:
        ``(A, b)`` with ``A.shape == (2, n_dof)`` and ``b.shape == (2,)``.
    """
    Jp = jacobian[:3]                     # (3, n_dof)
    grad = state.normal                   # use unit normal (= grad/‖grad‖)
    # n^T J_p  ∈ R^{1 × n_dof}
    nTJ = (grad[..., None, :] @ Jp[None, ...]).squeeze(-2) if Jp.ndim == 3 \
        else _matmul_row(grad, Jp)
    kd = stiffness * state.penetration
    # f_max:  -b (n^T J) u ≤ f_max - k d
    A_upper = -damping * nTJ
    b_upper = f_max - kd
    # f_min:   b (n^T J) u ≤ kd - f_min
    A_lower = damping * nTJ
    b_lower = kd - f_min
    A = _stack(A_upper, A_lower)
    b = _stack(b_upper, b_lower)
    return A, b


def penetration_bound_rows(
    state: ContactState,
    jacobian: Any,
    *,
    phi_target: float = 0.0,
    alpha: float = 1.0,
    side: str = "both",
) -> Tuple[Any, Any]:
    """CBF rows enforcing ``φ`` near a target via linearised dynamics.

    For a CBF ``h(p) = φ(p) - φ_target`` with extended class-K ``α h``:

        ḣ = ∇φ · v_tool = ∇φ · J_p u ≥ -α (φ - φ_target)

    Rearranged as ``A u ≤ b``:

        -(∇φ · J_p) u ≤ α (φ - φ_target).

    For a two-sided bound (``φ ∈ [φ_target - δ, φ_target + δ]``) call this
    twice with the appropriate sign.

    Args:
        state: ``ContactState`` (uses ``state.normal`` as the unit ``∇φ``).
        jacobian: ``(6, n_dof)`` or ``(3, n_dof)``; top three rows used.
        phi_target: desired SDF value at the tool tip.
        alpha: CBF gain (extended class-K coefficient).
        side: ``"upper"`` (φ ≤ target), ``"lower"`` (φ ≥ target), or
            ``"both"`` (default — emit both rows).

    Returns:
        ``(A, b)`` shaped ``(k, n_dof)`` and ``(k,)`` with ``k ∈ {1, 2}``.
    """
    if side not in ("upper", "lower", "both"):
        raise ValueError(f"side must be upper/lower/both, got {side!r}")
    Jp = jacobian[:3]
    grad = state.normal
    nTJ = _matmul_row(grad, Jp)
    # Surrogate SDF value: φ ≈ -penetration when in contact, else uses sign
    # via state.in_contact gate. We approximate φ from penetration so
    # callers don't need to thread the raw SDF query result through.
    phi_b = -state.penetration  # 0 outside contact; negative inside
    rows_A, rows_b = [], []
    if side in ("upper", "both"):
        rows_A.append(nTJ)
        rows_b.append(alpha * (phi_target - phi_b))
    if side in ("lower", "both"):
        rows_A.append(-nTJ)
        rows_b.append(alpha * (phi_b - phi_target))
    A = _stack(*rows_A)
    b = _stack(*rows_b)
    return A, b


# ---------- backend-agnostic helpers ---------------------------------------


def _matmul_row(vec: Any, mat: Any) -> Any:
    """``vec @ mat`` with ``vec`` shape ``(3,)`` and ``mat`` ``(3, n)``."""
    # numpy / jax / torch all support __matmul__ with these shapes.
    return vec @ mat


def _stack(*rows: Any) -> Any:
    """Stack rows along axis 0 (works for numpy, jax, torch)."""
    if len(rows) == 1:
        return rows[0][None, ...] if rows[0].ndim == 1 else rows[0]
    if isinstance(rows[0], np.ndarray):
        return np.stack(rows, axis=0)
    # JAX / torch
    try:
        import jax.numpy as jnp
        if isinstance(rows[0], jnp.ndarray):
            return jnp.stack(rows, axis=0)
    except Exception:
        pass
    try:
        import torch
        if isinstance(rows[0], torch.Tensor):
            return torch.stack(rows, dim=0)
    except Exception:
        pass
    return np.stack([np.asarray(r) for r in rows], axis=0)
