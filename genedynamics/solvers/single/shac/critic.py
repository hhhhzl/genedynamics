"""Pure-JAX MLP critic + hand-rolled Adam.

Why hand-rolled?
- optax is the right choice in production but introduces a runtime dep this
  repo doesn't currently require. The full Adam update is ~12 lines so we
  inline it with the same step API (`init`, `update`) optax exposes. If the
  user installs optax later, switching is a one-line change in shac_jax.py.

The critic is a small tanh MLP with Glorot init. Returns a scalar V(obs).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


# ---------------------------------------------------------------------------
# MLP (pure JAX, no flax)
# ---------------------------------------------------------------------------


def init_mlp(
    rng: jax.Array,
    in_dim: int,
    hidden: Tuple[int, ...],
    out_dim: int = 1,
) -> List[Dict[str, jnp.ndarray]]:
    """Glorot-initialized MLP. Layers are tanh except the final linear head."""
    sizes = [int(in_dim)] + [int(h) for h in hidden] + [int(out_dim)]
    keys = jax.random.split(rng, len(sizes) - 1)
    params: List[Dict[str, jnp.ndarray]] = []
    for k, in_d, out_d in zip(keys, sizes[:-1], sizes[1:]):
        scale = jnp.sqrt(jnp.asarray(2.0 / max(in_d, 1), dtype=jnp.float32))
        W = jax.random.normal(k, (in_d, out_d), dtype=jnp.float32) * scale
        b = jnp.zeros((out_d,), dtype=jnp.float32)
        params.append({"W": W, "b": b})
    return params


def mlp_apply(params: List[Dict[str, jnp.ndarray]], x: jnp.ndarray) -> jnp.ndarray:
    """Forward pass; supports leading-batch broadcasting (..., in_dim)."""
    h = x
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["W"] + layer["b"])
    out = h @ params[-1]["W"] + params[-1]["b"]
    return out


# ---------------------------------------------------------------------------
# Adam optimizer (drop-in init/update API)
# ---------------------------------------------------------------------------


def adam_init(params) -> Dict[str, Any]:
    """Init optimizer state. ``params`` must be a pytree of jnp.ndarray."""
    zero = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zero, "v": jax.tree_util.tree_map(jnp.zeros_like, params), "t": jnp.asarray(0, dtype=jnp.int32)}


def adam_update(
    grads,
    state: Dict[str, Any],
    params,
    lr: float,
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    grad_clip: float = 0.0,
) -> Tuple[Any, Dict[str, Any]]:
    """One Adam step. Returns (new_params, new_state).

    grad_clip > 0 → global L2 norm clip on gradients before the moment update.
    """
    if grad_clip > 0.0:
        flat, _ = ravel_pytree(grads)
        gnorm = jnp.linalg.norm(flat)
        clip_factor = jnp.minimum(jnp.asarray(1.0), grad_clip / (gnorm + 1e-12))
        grads = jax.tree_util.tree_map(lambda g: g * clip_factor, grads)

    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda mi, gi: beta1 * mi + (1 - beta1) * gi, state["m"], grads)
    v = jax.tree_util.tree_map(lambda vi, gi: beta2 * vi + (1 - beta2) * (gi * gi), state["v"], grads)
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    m_hat = jax.tree_util.tree_map(lambda mi: mi / bc1, m)
    v_hat = jax.tree_util.tree_map(lambda vi: vi / bc2, v)
    new_params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
        params, m_hat, v_hat,
    )
    return new_params, {"m": m, "v": v, "t": t}


def grad_norm(grads) -> jnp.ndarray:
    """Global L2 norm of a pytree of gradients (diagnostic)."""
    flat, _ = ravel_pytree(grads)
    return jnp.linalg.norm(flat)


# ---------------------------------------------------------------------------
# Polyak (target network)
# ---------------------------------------------------------------------------


def polyak_update(target, online, alpha: float):
    """target ← α * target + (1-α) * online (elementwise on pytree)."""
    return jax.tree_util.tree_map(
        lambda t, o: alpha * t + (1.0 - alpha) * o, target, online,
    )


# ---------------------------------------------------------------------------
# Discounted-sum / td-λ helpers
# ---------------------------------------------------------------------------


def discounted_sum(rewards: jnp.ndarray, discount: float) -> jnp.ndarray:
    """Σ_{t=0}^{h-1} γ^t r_t for a (h,) reward sequence."""
    h = rewards.shape[0]
    gammas = discount ** jnp.arange(h, dtype=rewards.dtype)
    return jnp.sum(gammas * rewards)


def td_lambda_targets(
    rewards: jnp.ndarray,        # (h,)
    values: jnp.ndarray,         # (h+1,) — V(s_0), V(s_1), ..., V(s_h)
    discount: float,
    lam: float,
) -> jnp.ndarray:
    """Compute td-λ value targets for s_0..s_{h-1}. Returns (h,) array.

    Recurrence (Sutton & Barto, eq. 12.18 generalized for td-λ):
        target_{h-1} = r_{h-1} + γ * V(s_h)
        target_t     = r_t + γ * [(1-λ) * V(s_{t+1}) + λ * target_{t+1}]   for t < h-1
    """
    h = rewards.shape[0]
    target_term = rewards[h - 1] + discount * values[h]

    def step(target_next, t_rev_idx):
        # We iterate from t = h-2 down to t = 0 (h-1 steps total).
        t = h - 2 - t_rev_idx
        v_next = values[t + 1]
        target_t = rewards[t] + discount * ((1 - lam) * v_next + lam * target_next)
        return target_t, target_t

    _, scan_out = jax.lax.scan(step, target_term, jnp.arange(h - 1, dtype=jnp.int32))
    # scan_out is [targets[h-2], targets[h-3], ..., targets[0]] — reverse it
    # and append the precomputed terminal target.
    targets = jnp.concatenate([scan_out[::-1], target_term[None]])
    return targets
