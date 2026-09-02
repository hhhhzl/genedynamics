"""MGA CFS retraction — wired exactly like 2GO.

2GO does ``CfsRetraction(backend="jax", filter_fn=inner._filter_actions_single_jit)``
(twogo_jax.py:528) and calls ``retract(state, traj, {"sched_state", "sched_params"})``.
MGA reuses the SAME upstream genemetry ``CfsRetraction`` the same way — only the
``filter_fn`` differs: instead of cfsmbd's SDF-obstacle CFS-QP filter, MGA's
filter projects the node-control trajectory onto the env's CLEAN-STATE constraint
manifold ``{C(U)=0}`` via a linearized (Gauss-Newton) CFS step
(eq:linearized_retraction):

    dU = - J_C^T (J_C J_C^T + reg I)^{-1} C(U),   U <- U + gain * dU

``C = env.manifold_residual(state, U)`` is analytic (no mjx rollout, §7.4), so
``J_C`` is an autodiff Jacobian with no physics backprop. The filter_fn signature
matches `CfsRetractionJax` exactly: ``(state, trajectory, sched_state,
sched_params) -> filtered_trajectory``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from genedynamics.genemetry.retraction.cfs import CfsRetraction


ResidualFn = Callable[[Any, Any, Any], Any]


def make_mga_cfs_filter(
    env: Any,
    *,
    residual_fn: Optional[ResidualFn] = None,
    n_iters: int = 1,
    reg: float = 1e-6,
    gain: float = 1.0,
):
    """CFS filter_fn for MGA: linearized projection onto ``env.manifold_residual``.
    Same call shape as cfsmbd's ``_filter_actions_single_jit``.

    ``residual_fn`` is the architecture seam for horizon-aware geometry. It
    receives ``(state, trajectory, t0)``; the default adapts the legacy env hook
    and preserves previous MGA behavior exactly.
    """
    residual = residual_fn or (
        lambda state, trajectory, t0: env.manifold_residual(state, trajectory)
    )

    def filter_fn(state, trajectory, sched_state, sched_params):
        shape = trajectory.shape
        t0 = (sched_params or {}).get("t0", 0.0)

        def C(u):
            return residual(state, u.reshape(shape), t0)

        U = trajectory.reshape(-1)
        for _ in range(int(n_iters)):
            c = C(U)
            if c.shape[0] == 0:                          # no constraint -> identity
                break
            J = jax.jacrev(C)(U)                         # (n_C, D) analytic
            JJt = J @ J.T + reg * jnp.eye(J.shape[0], dtype=U.dtype)
            U = U - gain * (J.T @ jnp.linalg.solve(JJt, c))
        return U.reshape(shape)

    return filter_fn


def make_mga_retraction(
    env: Any,
    *,
    residual_fn: Optional[ResidualFn] = None,
    n_iters: int = 1,
    reg: float = 1e-6,
    gain: float = 1.0,
) -> CfsRetraction:
    """Upstream genemetry ``CfsRetraction`` (jax) wrapping the MGA CFS filter —
    constructed exactly as 2GO constructs its retraction."""
    return CfsRetraction(backend="jax",
                         filter_fn=make_mga_cfs_filter(
                             env, residual_fn=residual_fn,
                             n_iters=n_iters, reg=reg, gain=gain,
                         ))


__all__ = ["ResidualFn", "make_mga_cfs_filter", "make_mga_retraction"]
