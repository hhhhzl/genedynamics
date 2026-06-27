"""JAX backend for ATACOM — tangent-space action transform (FAITHFUL reproduction).

Re-implements ATACOM (`baselines/rl_on_manifold/atacom/atacom.py`, mushroom_rl + torch) on
the jax stack, matching the vendored ``step_action_function`` (atacom.py:123-139) +
``pinv_null`` (utils/null_space_coordinate.py:8-26) + slack (`_compute_slack_variables`,
`_construct_Jc_psi`, `_compute_error_correction`):

  Jc, Jc⁻¹, Nc = pinv_null(Jc)          (Nc = null-space tangent basis of the augmented Jacobian)
  [Δu; ṡ] = Nc·α  −  Jc⁻¹·(K_c · C)      (act_b tangent action + act_err drift to the manifold)
  u = Δu,   s ← s + ṡ·dt                  (ψ drift term ≈ 0 for our algebraic, per-step constraint)

adapted to the arm: the controllable variable is the action ``u`` (dim ``nu``); the EQUALITY
constraint ``f`` is the env clean-state manifold ``env.manifold_residual`` (the SAME manifold
MDAC uses, dim ``n_f``); the INEQUALITY ``g`` is ``env.constraint_residual``'s ``g`` (force
bounds, dim ``n_g``), turned into an equality via a slack ``s`` (``g + ½s² = 0``, slack columns
``diag(s)`` in ``Jc``). The RL policy ``α`` is the TANGENT dimension ``null = nu − n_f`` and is
trained ON the manifold (the wrapper applies this transform inside ``env.step`` during BOTH
training and deploy). ``pinv_null``'s dynamic-rank ``rref`` basis is replaced by the SVD
null-space (a valid tangent basis the policy adapts to); the pinv is rcond-truncated.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp

_N_F = 3       # arm equality manifold dim (manifold_residual: ξ, η, force)
_N_G = 2       # arm inequality dim (force bounds)


def atacom_null_dim(env: Any) -> int:
    """Tangent (policy) dimension = nu − n_f (atacom.py:39 dims['null'])."""
    return int(env.action_size) - _N_F


def _pinv_null(A: jnp.ndarray, n_c: int, rcond: float = 1e-6):
    """``pinv_null`` (SVD): A⁺ (rcond-truncated) and the null-space basis Nc (the trailing
    right-singular vectors). Assumes the augmented Jacobian has ``n_c`` rows."""
    u, s, vh = jnp.linalg.svd(A, full_matrices=True)          # u(n_c,n_c) s(n_c,) vh(N,N)
    sinv = jnp.where(s > rcond * jnp.max(s), 1.0 / s, 0.0)    # rcond-truncated inverse svals
    A_pinv = (vh[:n_c, :].T * sinv[None, :]) @ u.T            # (N, n_c)
    Nc = vh[n_c:, :].T                                        # (N, N − n_c) null space
    return A_pinv, Nc


def make_atacom_transform(env: Any, *, Kc: float = 1.0, time_step: float = 0.02,
                          action_limit: float = 1.0) -> Callable[[Any, Any, Any], Tuple[Any, Any]]:
    """Return ``(state, alpha, s) -> (u, s_new)``: the ATACOM tangent-space action transform."""
    nu, n_c = int(env.action_size), _N_F + _N_G
    K_c = jnp.full((n_c,), float(Kc))

    def _CofU(state, u):                                       # equality f + inequality g at action u
        f = env.manifold_residual(state, u[None])             # (n_f,)
        g = env.constraint_residual(state, u)[1]              # (n_g,)
        return jnp.concatenate([f, g])                       # (n_c,)

    def transform(state, alpha, s):
        u_ref = jnp.zeros((nu,), jnp.float32)
        Cu = _CofU(state, u_ref)                              # C at u=0 (no slack term)
        C = Cu + jnp.concatenate([jnp.zeros((_N_F,)), 0.5 * s ** 2])   # + ½s² on the g rows
        Ju = jax.jacobian(lambda u: _CofU(state, u))(u_ref)  # (n_c, nu)  ∂C/∂u
        Js = jnp.concatenate([jnp.zeros((_N_F, _N_G)), jnp.diag(s)], axis=0)   # (n_c, n_g)  ∂C/∂s
        Jc = jnp.concatenate([Ju, Js], axis=1)               # (n_c, nu + n_g)
        Jc_inv, Nc = _pinv_null(Jc, n_c)                     # (nu+n_g, n_c), (nu+n_g, null_dim)
        duds = Nc @ alpha - Jc_inv @ (K_c * C)               # act_b (tangent) + act_err (drift)
        u = jnp.clip(duds[:nu], -action_limit, action_limit)
        s_new = jnp.maximum(s + duds[nu:nu + _N_G] * time_step, 0.0)   # slack >= 0
        return u, s_new

    return transform


def init_slack(env: Any, state: Any) -> jnp.ndarray:
    """Initial slack ``s = sqrt(max(-2 g, 0))`` (atacom.py:_compute_slack_variables) at u=0."""
    g = env.constraint_residual(state, jnp.zeros((int(env.action_size),)))[1]
    return jnp.sqrt(jnp.maximum(-2.0 * g, 0.0))


__all__ = ["make_atacom_transform", "atacom_null_dim", "init_slack"]
