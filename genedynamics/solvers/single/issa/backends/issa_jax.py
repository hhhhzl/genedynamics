"""JAX backend for ISSA — AdamBA safe-action projection (FAITHFUL reproduction).

Re-implements the AdamBA (Adaptive Momentum Boundary Approximation) safe-set projection of
ISSA (`baselines/Implicit_Safe_Set_Algorithm/toy_problem/{AdamBA,adamba_ssa_project_distance}.py`,
TF1 + safety-gym) on the jax stack, matching the vendored algorithm:

  * DERIVATIVE-FREE ray search: from the unsafe action, sample random unit directions; along
    each ray run AdamBA's exponential-expand → bisection to the SAFE-SET BOUNDARY
    (`AdamBA.py:90-122`, the exact ``eta`` schedule: ×2 to outreach while unsafe; on first
    crossing ×0.25 then ×0.5 bisection, stepping back when safe / forward when unsafe).
  * Safe set = ``{u : g(state,u) ≤ 0}`` (the env inequality residual — ISSA's "safe set
    where the safety index does not increase"); ``unsafe = max_i g_i > 0`` (`utils.chk_unsafe`).
  * Projection = the QP ``min ||u'−u||² s.t. u' ∈ safe set`` (`adamba_ssa_project_distance.py:
    37-47`); its N-D realization here is the CLOSEST safe boundary point found by the rays
    (minimal-change projection onto the safe-set boundary). NO gradient of the constraint.

The ray search is vectorized over directions and runs a fixed iteration budget (jittable);
the safe/unsafe gate is a deploy-time Python branch (the controller loop is not jitted).
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp


def make_issa_projection(env: Any, *, n_dirs: int = 20, n_iters: int = 50, bound: float = 1e-4,
                         action_limit: float = 1.0, seed: int = 0) -> Callable[[Any, Any], Any]:
    """Return ``(state, action) -> action`` projecting onto the safe set via AdamBA ray search."""
    def _g(state, u):
        return env.constraint_residual(state, u)[1]            # inequality residual g (safe: g<=0)

    gmax = jax.jit(lambda state, u: jnp.max(_g(state, u)))

    def _adamba(state, u, key):
        nu = u.shape[0]
        dirs = jax.random.normal(key, (n_dirs, nu))
        dirs = dirs / (jnp.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)
        unsafe_one = lambda uu: (jnp.max(_g(state, uu)) > 0.0).astype(jnp.float32)

        def body(carry, _):
            u_cur, eta, dflag, valid = carry
            flag = jax.vmap(unsafe_one)(u_cur)                 # (N,) 1 = unsafe
            oob = jnp.any(jnp.abs(u_cur) > action_limit, axis=1)
            valid = valid & (~oob)
            expand = (flag > 0.5) & (dflag < 0.5)              # outreach: u += eta·dir; eta×2
            start = (flag < 0.5) & (dflag < 0.5)               # first crossing: eta×0.25, start refine
            ref_u = (flag > 0.5) & (dflag > 0.5)               # refine, unsafe: u += eta·dir; eta×0.5
            ref_s = (flag < 0.5) & (dflag > 0.5)               # refine, safe: u −= eta·dir; eta×0.5
            move = (jnp.where(expand | ref_u, eta, 0.0) - jnp.where(ref_s, eta, 0.0))[:, None] * dirs
            u_cur = u_cur + move
            eta = eta * jnp.where(expand, 2.0, jnp.where(start, 0.25, 0.5))
            dflag = jnp.where(start, 1.0, dflag)
            return (u_cur, eta, dflag, valid), None

        u0 = jnp.broadcast_to(u, (n_dirs, nu))
        init = (u0, jnp.full((n_dirs,), bound), jnp.zeros((n_dirs,)), jnp.ones((n_dirs,), bool))
        (u_cur, _eta, _df, valid), _ = jax.lax.scan(body, init, None, length=int(n_iters))

        # keep boundary points that ended SAFE and in-bounds; pick the closest to u (min-‖·‖ QP).
        ok = valid & (jax.vmap(unsafe_one)(u_cur) < 0.5)
        u_cur = jnp.clip(u_cur, -action_limit, action_limit)
        dist = jnp.where(ok, jnp.sum((u_cur - u) ** 2, axis=1), jnp.inf)
        best = jnp.argmin(dist)
        # fall back to the clipped nominal action if no ray found a safe boundary point.
        return jnp.where(jnp.isfinite(dist[best]), u_cur[best], jnp.clip(u, -action_limit, action_limit))

    adamba = jax.jit(_adamba)
    key0 = jax.random.PRNGKey(int(seed))

    def _proj(state, a):
        a = jnp.clip(jnp.asarray(a), -action_limit, action_limit)
        if float(gmax(state, a)) <= 0.0:                       # already safe -> identity
            return a
        return adamba(state, a, key0)

    return _proj


__all__ = ["make_issa_projection"]
