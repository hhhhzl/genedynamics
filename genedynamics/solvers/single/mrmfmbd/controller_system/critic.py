"""Short-horizon value function V_psi for the in-loop SHAC critic (Stage 5b;
new_version.txt eq:shac_q). A small MLP over compact state features
[com_x(3), com_v(3), e_x(d_e)]; trained ONLINE by TD(lambda) inside the diffusion
scan (params carried + updated each reverse step). It bootstraps the
short-horizon return so the controller-latent refinement is less myopic:

    Qhat_h(c) = sum_{t<h} gamma^t r_t + gamma^h V_psi(s_h).

critic_enabled=False -> V_psi is unused (Qhat = pure h-step return), i.e. exact
Stage-5b behavior.
"""
from __future__ import annotations

import math
from typing import Dict

import jax
import jax.numpy as jnp

CriticParams = Dict[str, jnp.ndarray]


def _glorot(key, shape):
    lim = math.sqrt(6.0 / float(sum(shape)))
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim, dtype=jnp.float32)


def state_dim(d_e: int) -> int:
    """com_x(3) + com_v(3) + e_x(d_e)."""
    return 6 + int(d_e)


def init_critic(seed: int, d_state: int, d_hidden: int = 32) -> CriticParams:
    ks = jax.random.split(jax.random.PRNGKey(int(seed) + 991), 2)
    return {
        "w1": _glorot(ks[0], (int(d_state), int(d_hidden))),
        "b1": jnp.zeros((int(d_hidden),), jnp.float32),
        "w2": _glorot(ks[1], (int(d_hidden), 1)) * 0.1,   # small last layer -> V~0 at init
        "b2": jnp.zeros((1,), jnp.float32),
    }


def critic_value(params: CriticParams, s: jnp.ndarray) -> jnp.ndarray:
    """s (..., d_state) -> value (...,)."""
    h = jnp.tanh(s @ params["w1"] + params["b1"])
    return (h @ params["w2"] + params["b2"])[..., 0]


def lambda_returns(rewards: jnp.ndarray, values_next: jnp.ndarray,
                   gamma: float, lam: float) -> jnp.ndarray:
    """TD(lambda) returns for a single trajectory.

    rewards      : (h,)  r_0..r_{h-1}
    values_next  : (h,)  V(s_{t+1}) (bootstrap value after step t)
    Returns G (h,):  G_t = r_t + gamma*((1-lam)*V(s_{t+1}) + lam*G_{t+1}).
    """
    h = rewards.shape[0]

    def body(G_next, idx):
        t = h - 1 - idx
        G = rewards[t] + gamma * ((1.0 - lam) * values_next[t] + lam * G_next)
        return G, G

    _, G_rev = jax.lax.scan(body, values_next[h - 1], jnp.arange(h))
    return jnp.flip(G_rev, axis=0)


def critic_td_loss(params: CriticParams, states: jnp.ndarray, rewards: jnp.ndarray,
                   gamma: float, lam: float) -> jnp.ndarray:
    """Mean-squared TD(lambda) loss over a batch of trajectories.

    states  : (B, h, d_state)   post-step states s_1..s_h
    rewards : (B, h)
    """
    def _one(s_traj, r_traj):
        v = critic_value(params, s_traj)                       # (h,) = V(s_1..s_h)
        v_next = jnp.concatenate([v[1:], v[-1:]])              # bootstrap last with itself
        G = lambda_returns(r_traj, jax.lax.stop_gradient(v_next), gamma, lam)
        return jnp.mean((v - jax.lax.stop_gradient(G)) ** 2)

    return jnp.mean(jax.vmap(_one)(states, rewards))


def critic_sgd_step(params: CriticParams, states: jnp.ndarray, rewards: jnp.ndarray,
                    gamma: float, lam: float, lr: float) -> CriticParams:
    """One SGD step on the TD(lambda) loss (carried across diffusion steps)."""
    g = jax.grad(critic_td_loss)(params, states, rewards, gamma, lam)
    return {k: params[k] - lr * jnp.nan_to_num(g[k]) for k in params}
