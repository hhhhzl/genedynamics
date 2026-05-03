"""SHAC policy loss + per-episode training step.

The actor minimizes::

    L_phi = - (1 / (N*h)) Σ_i [ Σ_{t=0}^{h-1} γ^t r_{i,t}  +  γ^h V_target(s_{i,h}) ]

evaluated on N short rollouts of length h sampled in parallel from saved end-
states of the previous episode (Xu et al., ICLR 2022, Eq. 5).

This module exposes the loss + a single jit'd `episode_step` that does the
full SHAC update (actor grad, critic SGD, target Polyak). The high-level
SHACSolver in shac.py just calls this in a Python loop.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp

from .critic import (
    mlp_apply,
    adam_update,
    polyak_update,
    td_lambda_targets,
    grad_norm,
)


# ---------------------------------------------------------------------------
# Actor loss
# ---------------------------------------------------------------------------


def make_actor_loss_fn(rollout_fn: Callable, h: int, discount: float):
    """Build a jittable actor-loss closure parameterized by ``rollout_fn``.

    ``rollout_fn`` must have the signature::

        rollout_fn(carry_init, t0, phi) -> (final_carry, rewards (h,), obs (h+1, D))

    where ``phi`` is shape (phi_dim,) and the return arrays are jnp arrays.
    """
    gammas = jnp.power(jnp.asarray(discount, dtype=jnp.float32),
                        jnp.arange(int(h), dtype=jnp.float32))   # (h,)
    discount_h = jnp.asarray(discount ** int(h), dtype=jnp.float32)

    def actor_loss(phi_mean, sigma, eps, carries_init, t0, critic_target_params):
        # eps: (N, phi_dim). Reparameterized stochastic policy: phi_i = mu + sigma * eps_i.
        phis = phi_mean[None, :] + sigma * eps                  # (N, phi_dim)

        def per_env(phi_i, carry_i):
            return rollout_fn(carry_i, t0, phi_i)

        final_carries, rewards, obs = jax.vmap(per_env)(phis, carries_init)
        # rewards: (N, h), obs: (N, h+1, D)
        # Discounted reward sum per env.
        rsum = jnp.sum(gammas[None, :] * rewards, axis=-1)      # (N,)
        # Terminal value (uses TARGET critic).
        v_term = mlp_apply(critic_target_params, obs[:, -1, :]).squeeze(-1)   # (N,)
        L = -jnp.mean(rsum + discount_h * v_term)
        return L, (final_carries, rewards, obs, phis)

    return actor_loss


# ---------------------------------------------------------------------------
# Critic SGD on td-λ targets
# ---------------------------------------------------------------------------


def critic_sgd_update(
    critic_params,
    critic_state: Dict[str, Any],
    target_params,
    obs_flat: jnp.ndarray,        # (N*h, D) — observations for s_0..s_{h-1}
    target_values: jnp.ndarray,   # (N*h,)  — td-λ targets, treated as constant
    n_iters: int,
    n_minibatches: int,
    lr: float,
    grad_clip: float,
    rng: jax.Array,
) -> Tuple[Any, Dict[str, Any], jnp.ndarray]:
    """SGD on the critic against precomputed td-λ targets.

    Returns updated (params, opt_state, mean_critic_loss).
    """
    n = obs_flat.shape[0]
    mb = max(n // max(int(n_minibatches), 1), 1)
    losses = []
    rng_curr = rng
    for _ in range(int(n_iters)):
        rng_curr, perm_key = jax.random.split(rng_curr)
        perm = jax.random.permutation(perm_key, n)
        obs_p = obs_flat[perm]
        tgt_p = target_values[perm]
        for k in range(0, n, mb):
            ob = obs_p[k:k + mb]
            tg = tgt_p[k:k + mb]

            def loss_fn(p):
                pred = mlp_apply(p, ob).squeeze(-1)
                return jnp.mean((pred - tg) ** 2)

            l, g = jax.value_and_grad(loss_fn)(critic_params)
            critic_params, critic_state = adam_update(
                g, critic_state, critic_params, lr=lr, grad_clip=grad_clip,
            )
            losses.append(l)
    mean_loss = jnp.mean(jnp.stack(losses)) if losses else jnp.float32(0.0)
    return critic_params, critic_state, mean_loss


# ---------------------------------------------------------------------------
# One full SHAC episode (actor grad + critic SGD + Polyak)
# ---------------------------------------------------------------------------


def episode_step(
    *,
    phi_mean: jnp.ndarray,
    actor_state: Dict[str, Any],
    critic_params,
    critic_state: Dict[str, Any],
    target_params,
    carries_init,
    t0: jnp.ndarray,
    rng: jax.Array,
    actor_loss_fn: Callable,
    actor_lr: float,
    actor_grad_clip: float,
    critic_lr: float,
    critic_grad_clip: float,
    n_critic_iters: int,
    n_critic_minibatches: int,
    target_alpha: float,
    sigma: float,
    discount: float,
    td_lambda_coef: float,
) -> Dict[str, Any]:
    """One outer SHAC episode.

    Returns a dict with: phi_mean, actor_state, critic_params, critic_state,
    target_params, final_carries, mean_episode_return, actor_loss,
    critic_loss, actor_grad_norm, rng.
    """
    rng, eps_key, critic_key = jax.random.split(rng, 3)

    # Sample reparam noise.
    n_envs = carries_init[0].shape[0]
    phi_dim = phi_mean.shape[0]
    eps = jax.random.normal(eps_key, (n_envs, phi_dim), dtype=jnp.float32)

    # Actor grad.
    (loss_value, aux), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
        phi_mean, jnp.asarray(sigma, dtype=jnp.float32), eps,
        carries_init, t0, target_params,
    )
    final_carries, rewards, obs, phis = aux

    g_norm = jnp.linalg.norm(grads)
    phi_mean, actor_state = adam_update(
        grads, actor_state, phi_mean, lr=actor_lr, grad_clip=actor_grad_clip,
    )

    # Critic targets via td-λ on the trajectories we just collected.
    def per_env_targets(rewards_i, obs_i):
        # values: V_target(s_0), ..., V_target(s_h) — (h+1,)
        values = mlp_apply(target_params, obs_i).squeeze(-1)
        return td_lambda_targets(rewards_i, values, discount, td_lambda_coef)

    targets = jax.vmap(per_env_targets)(rewards, obs)        # (N, h)
    targets_flat = targets.reshape(-1)                       # (N*h,)
    obs_for_critic = obs[:, :-1, :].reshape(-1, obs.shape[-1])

    critic_params, critic_state, critic_loss = critic_sgd_update(
        critic_params, critic_state, target_params,
        obs_for_critic, jax.lax.stop_gradient(targets_flat),
        n_iters=n_critic_iters,
        n_minibatches=n_critic_minibatches,
        lr=critic_lr,
        grad_clip=critic_grad_clip,
        rng=critic_key,
    )

    # Polyak target update.
    target_params = polyak_update(target_params, critic_params, target_alpha)

    return {
        "phi_mean": phi_mean,
        "actor_state": actor_state,
        "critic_params": critic_params,
        "critic_state": critic_state,
        "target_params": target_params,
        "final_carries": final_carries,
        "mean_episode_return": jnp.mean(jnp.sum(rewards, axis=-1)),
        "actor_loss": loss_value,
        "critic_loss": critic_loss,
        "actor_grad_norm": g_norm,
        "rng": rng,
    }
