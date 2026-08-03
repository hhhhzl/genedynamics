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
        g = jnp.asarray(env.constraint_residual(state, u)[1]).reshape((n_g,))
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
        Ju = jax.jacobian(lambda u: constraint_at(state, u))(u_ref)
        Js = jnp.concatenate([
            jnp.zeros((n_f, n_g), dtype=alpha.dtype),
            jnp.diag(slack),
        ], axis=0)
        Jc = jnp.concatenate([Ju, Js], axis=1)
        Jc_inv, Nc = _pinv_null(Jc, n_c, rcond=rcond)
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
    g = jnp.asarray(env.constraint_residual(
        state, jnp.zeros((int(env.action_size),), dtype=jnp.float32)
    )[1]).reshape((n_g,))
    return jnp.sqrt(jnp.maximum(-2.0 * g, 0.0))


__all__ = [
    "make_atacom_transform", "atacom_constraint_dims", "atacom_null_dim",
    "init_slack", "_pinv_null",
]
