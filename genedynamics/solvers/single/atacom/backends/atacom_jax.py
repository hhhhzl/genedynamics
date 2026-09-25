"""Task-adapted JAX ATACOM tangent-space action transform.

The original ATACOM controller operates on generalized accelerations.  The arm
comparison has a task primitive instead, so its task-owned contract exposes an
algebraic equality manifold ``f(state, u)=0`` and inequalities ``g(state,u)<=0``.
The ATACOM augmentation is otherwise unchanged: inequalities become
``g + 1/2 s^2 = 0`` and the policy controls the null space of the augmented
Jacobian while a pseudoinverse term corrects manifold drift.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp


def _constraint_residual(env: Any, state: Any, action: jnp.ndarray):
    """Return the constraint chart owned by an ATACOM-capable task.

    The generic contract remains ``constraint_residual`` so existing surface
    scanning and humanoid tasks keep their exact behavior.  Contact tasks may
    instead expose ``atacom_constraint_residual`` when their safety variables
    are state quantities whose response to the action must be obtained through
    the known dynamics.  This mirrors ATACOM's requirement that the constraint
    Jacobian be taken through the controllable dynamics rather than treating a
    measured state constraint as an action-independent algebraic row.
    """
    hook = getattr(env, "atacom_constraint_residual", None)
    if hook is not None:
        return hook(state, action)
    # The legacy/general task contract may expose additional equality
    # diagnostics whose width is unrelated to ATACOM's action manifold.  Only
    # an explicit ATACOM hook may replace ``manifold_residual``; fallback tasks
    # retain their exact pre-existing equality chart and contribute only g.
    _, g = env.constraint_residual(state, action)
    return jnp.zeros((0,), dtype=jnp.asarray(action).dtype), g


def atacom_constraint_dims(env: Any) -> Tuple[int, int]:
    """Read and validate the task-owned equality/inequality dimensions."""
    if not hasattr(env, "manifold_constraint_size"):
        raise ValueError("ATACOM requires env.manifold_constraint_size")
    if not hasattr(env, "inequality_constraint_size"):
        raise ValueError("ATACOM requires env.inequality_constraint_size")
    n_f = int(env.manifold_constraint_size)
    n_g = int(env.inequality_constraint_size)
    nu = int(env.action_size)
    if n_f < 0 or n_g < 0 or n_f > nu:
        raise ValueError(
            f"invalid ATACOM dimensions: nu={nu}, n_f={n_f}, n_g={n_g}"
        )
    return n_f, n_g


def atacom_null_dim(env: Any) -> int:
    """Policy dimension ``nu - n_f`` from the environment contract."""
    n_f, _ = atacom_constraint_dims(env)
    return int(env.action_size) - n_f


def _pinv_null(A: jnp.ndarray, n_c: int, rcond: float = 1e-6):
    """Full-row-rank SVD pseudoinverse and fixed-size null-space basis."""
    u, s, vh = jnp.linalg.svd(A, full_matrices=True)
    scale = jnp.maximum(jnp.max(s), jnp.finfo(s.dtype).tiny)
    sinv = jnp.where(s > float(rcond) * scale, 1.0 / s, 0.0)
    A_pinv = (vh[:n_c, :].T * sinv[None, :]) @ u.T
    Nc = vh[n_c:, :].T
    return A_pinv, Nc


def _damped_qr_pinv_null(
    A: jnp.ndarray, n_c: int, rcond: float = 1e-6,
):
    """Finite fixed-width projection for structurally rank-deficient tasks.

    Some task contracts include measured-state inequalities whose instantaneous
    action Jacobian is exactly zero.  SVD is appropriate for the full-rank arm
    contracts, but XLA's SVD can return NaNs for that repeated-zero spectrum.
    A complete QR still supplies the preregistered ``n_var - n_c`` tangent
    basis, while a damped right inverse safely ignores uncontrollable rows.
    """
    A = jnp.nan_to_num(jnp.asarray(A), nan=0.0, posinf=1e6, neginf=-1e6)
    q, _ = jnp.linalg.qr(A.T, mode="complete")
    Nc = q[:, n_c:]
    gram = A @ A.T
    scale = jnp.maximum(jnp.max(jnp.diag(gram)), jnp.asarray(1.0, A.dtype))
    regularized = gram + (float(rcond) * scale) * jnp.eye(n_c, dtype=A.dtype)
    A_pinv = A.T @ jnp.linalg.solve(regularized, jnp.eye(n_c, dtype=A.dtype))
    return jnp.nan_to_num(A_pinv), jnp.nan_to_num(Nc)


def make_atacom_transform(env: Any, *, Kc: float = 1.0,
                          time_step: float = 0.02,
                          action_limit: float = 1.0,
                          rcond: float = 1e-6) -> Callable:
    """Return ``(state, alpha, slack) -> (u, slack_next)``."""
    nu = int(env.action_size)
    n_f, n_g = atacom_constraint_dims(env)
    n_c = n_f + n_g
    K_c = jnp.full((n_c,), float(Kc), dtype=jnp.float32)

    def constraint_at(state, u):
        f = jnp.asarray(env.manifold_residual(state, u[None])).reshape((n_f,))
        task_f, task_g = _constraint_residual(env, state, u)
        task_f = jnp.asarray(task_f).reshape((-1,))
        if task_f.size == n_f:
            # A task-specific ATACOM chart may supply a dynamics-aware
            # equality residual as well.  PegInsert deliberately keeps the
            # existing action manifold, while the hook remains complete for
            # future task-owned inverse-dynamics charts.
            f = task_f
        elif task_f.size != 0:
            raise ValueError(
                "ATACOM equality residual width must be zero or match "
                f"manifold_constraint_size: {task_f.size} != {n_f}"
            )
        g = jnp.asarray(task_g).reshape((n_g,))
        return jnp.concatenate([f, g])

    def transform(state, alpha, slack):
        alpha = jnp.asarray(alpha)
        if alpha.shape != (nu - n_f,):
            raise ValueError(
                f"ATACOM alpha shape {alpha.shape} != {(nu - n_f,)}"
            )
        u_ref = jnp.zeros((nu,), dtype=alpha.dtype)
        raw_c = constraint_at(state, u_ref)
        C = raw_c + jnp.concatenate([
            jnp.zeros((n_f,), dtype=alpha.dtype),
            0.5 * slack ** 2,
        ])
        # MJX's iterative contact solver contains dynamic ``while_loop``
        # primitives, for which reverse-mode differentiation is undefined.
        # ATACOM needs the small control Jacobian (13 columns for PegInsert),
        # so forward mode is both the valid differentiation mode and the
        # direct analogue of the analytic constraint Jacobian in the paper.
        Ju = jax.jacfwd(lambda u: constraint_at(state, u))(u_ref)
        Js = jnp.concatenate([
            jnp.zeros((n_f, n_g), dtype=alpha.dtype),
            jnp.diag(slack),
        ], axis=0)
        Jc = jnp.concatenate([Ju, Js], axis=1)
        projection_solver = str(
            getattr(env, "atacom_projection_solver", "svd")
        )
        if projection_solver == "damped_qr":
            Jc_inv, Nc = _damped_qr_pinv_null(Jc, n_c, rcond=rcond)
        elif projection_solver == "svd":
            Jc_inv, Nc = _pinv_null(Jc, n_c, rcond=rcond)
        else:
            raise ValueError(
                f"unknown ATACOM projection solver {projection_solver!r}"
            )
        duds = Nc @ alpha - Jc_inv @ (K_c * C)
        u = jnp.clip(duds[:nu], -action_limit, action_limit)
        # Upstream ATACOM does not clamp slack: its sign is immaterial to s^2
        # and is needed by the next augmented Jacobian.
        slack_next = slack + duds[nu:nu + n_g] * float(time_step)
        return u, slack_next

    return transform


def init_slack(env: Any, state: Any) -> jnp.ndarray:
    """Initialize ``s = sqrt(max(-2g, 0))`` as in upstream ATACOM."""
    _, n_g = atacom_constraint_dims(env)
    action = jnp.zeros((int(env.action_size),), dtype=jnp.float32)
    g = jnp.asarray(_constraint_residual(env, state, action)[1]).reshape((n_g,))
    return jnp.sqrt(jnp.maximum(-2.0 * g, 0.0))


__all__ = [
    "make_atacom_transform", "atacom_constraint_dims", "atacom_null_dim",
    "init_slack", "_pinv_null", "_damped_qr_pinv_null",
]
