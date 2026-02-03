"""
Fast batch-friendly prefix-sum coupled slack-QP solver (JAX).

Why this exists:
- The reference implementation lives under `enerdynamics.core.constraints.solvers`,
  but parts of that directory are not writable in this environment.
- We also want a batch/JIT-friendly implementation that avoids dynamic `while_loop`
  termination and supports `maxiter` being a traced scalar.

This function is intended to be monkey-patched onto:
`enerdynamics.core.constraints.solvers.jaxopt_osqp_solver.solve_slack_qp_prefixsum_jax`
so existing filters keep working without editing the core module.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp


def solve_slack_qp_prefixsum_jax(
    u_nom: jnp.ndarray,  # (H, act_dim)
    A_per_step: jnp.ndarray,  # (H, K, act_dim)
    b_per_step: jnp.ndarray,  # (H, K)
    rho: jnp.ndarray | float,
    *,
    control_limit: Optional[float] = None,
    tol: float = 1e-6,
    maxiter: int | jnp.ndarray = 40,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Structured slack-QP solver for prefix-sum coupled constraints (single-integrator style).

    Constraints:
        <a_{t,k}, cumsum(u)[t]> >= b_{t,k} - xi_{t,k},  xi_{t,k} >= 0

    Optimization:
    - Uses a fixed-iteration `lax.fori_loop` with masked early-stop, which is
      robust under batching/vmap and when `maxiter` is a traced JAX scalar.
    - Each iteration does ONE (cumsum + einsum + argmax) pass; no extra
      `max_violation(u_next)` recomputation inside the loop.
    """
    u_nom = jnp.asarray(u_nom, dtype=jnp.float32)
    A_per_step = jnp.asarray(A_per_step, dtype=jnp.float32)
    b_per_step = jnp.asarray(b_per_step, dtype=jnp.float32)

    H, K, act_dim = A_per_step.shape
    assert u_nom.shape == (H, act_dim)
    assert b_per_step.shape == (H, K)

    valid = jnp.isfinite(b_per_step)

    if control_limit is None:
        u0 = u_nom
        L_lo = -jnp.inf
        L_hi = jnp.inf
    else:
        L = jnp.asarray(float(control_limit), dtype=jnp.float32)
        L_lo = -L
        L_hi = L
        u0 = jnp.clip(u_nom, L_lo, L_hi)

    rho_eff = jnp.asarray(rho, dtype=jnp.float32)
    rho_eff = jnp.maximum(rho_eff, 1e-9)
    tol_j = jnp.asarray(float(tol), dtype=jnp.float32)

    # Fixed upper bound (not unrolled): safe and fast for JIT/vmap.
    MAXITER_STATIC = 1024
    try:
        # Fast path if maxiter is a Python int
        maxiter_i = int(maxiter)
        maxiter_i = max(0, min(maxiter_i, MAXITER_STATIC))
        maxiter_j = jnp.asarray(maxiter_i, dtype=jnp.int32)
    except Exception:
        maxiter_j = jnp.asarray(maxiter, dtype=jnp.int32)
        maxiter_j = jnp.clip(maxiter_j, 0, MAXITER_STATIC)

    time_idx = jnp.arange(H, dtype=jnp.int32)

    def max_violation(u: jnp.ndarray) -> jnp.ndarray:
        u_prefix = jnp.cumsum(u, axis=0)
        lhs = jnp.einsum("hkd,hd->hk", A_per_step, u_prefix)
        viol = jnp.maximum(0.0, jnp.where(valid, b_per_step - lhs, -jnp.inf))
        return jnp.max(viol)

    def body(i: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        u_prefix = jnp.cumsum(u, axis=0)
        lhs = jnp.einsum("hkd,hd->hk", A_per_step, u_prefix)
        viol_hk = jnp.maximum(0.0, jnp.where(valid, b_per_step - lhs, -jnp.inf))

        flat_idx = jnp.argmax(viol_hk.reshape(-1))
        t_idx = flat_idx // jnp.asarray(K, dtype=jnp.int32)
        k_idx = flat_idx - t_idx * jnp.asarray(K, dtype=jnp.int32)

        a = A_per_step[t_idx, k_idx]  # (act_dim,)
        v = viol_hk[t_idx, k_idx]

        den = (jnp.asarray(t_idx + 1, dtype=jnp.float32) * jnp.dot(a, a)) + 1e-9
        lam = v / (den + 1.0 / rho_eff)

        prefix_mask = (time_idx <= t_idx).astype(jnp.float32)  # (H,)
        u_next = u + prefix_mask[:, None] * (lam * a[None, :])
        u_next = jnp.clip(u_next, L_lo, L_hi)

        do_update = jnp.logical_and(i < maxiter_j, v > tol_j)
        return jax.lax.cond(do_update, lambda _: u_next, lambda _: u, operand=None)

    u_star = jax.lax.fori_loop(0, MAXITER_STATIC, body, u0)
    v_star = max_violation(u_star)
    return u_star, v_star

