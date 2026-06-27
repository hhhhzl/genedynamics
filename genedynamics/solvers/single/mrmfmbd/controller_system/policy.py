"""Pure-JAX policy + morphology-embedding primitives for the learned controller.

The forward pass that actually runs inside the differentiable MPM rollout lives
in `envs/external/jax_mpm/scene.py:compute_actuation_learned` (kept there so the
env has no dependency on the solver package). This module builds the parameters
(beta = policy MLP, E_chi = morphology projection) and exposes the same math for
host-side construction + unit tests.
"""
from __future__ import annotations

import math
from typing import Dict

import jax
import jax.numpy as jnp

ControllerParams = Dict[str, jnp.ndarray]


def time_encoding(env_t, env_horizon) -> jnp.ndarray:
    """psi(t): 2-dim periodic encoding of normalized episode time."""
    t = jnp.asarray(env_t, dtype=jnp.float32) / jnp.maximum(jnp.float32(env_horizon), 1.0)
    ang = 2.0 * math.pi * t
    return jnp.stack([jnp.sin(ang), jnp.cos(ang)])


def obs_dim(d_c: int, d_e: int, n_proprio: int = 1) -> int:
    """Observation width fed to the policy: [proprio, psi(2), e_x(d_e), c(d_c)]."""
    return int(n_proprio) + 2 + int(d_e) + int(d_c)


def _glorot(key, shape):
    lim = math.sqrt(6.0 / float(sum(shape)))
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim, dtype=jnp.float32)


def init_policy(seed: int, d_obs: int, d_hidden: int, n_act: int) -> ControllerParams:
    """Fixed-architecture 1-hidden-layer tanh MLP: obs -> (n_act,) in [-1, 1]."""
    ks = jax.random.split(jax.random.PRNGKey(int(seed)), 4)
    return {
        "w1": _glorot(ks[0], (int(d_obs), int(d_hidden))),
        "b1": jnp.zeros((int(d_hidden),), jnp.float32),
        "w2": _glorot(ks[1], (int(d_hidden), int(n_act))),
        "b2": jnp.zeros((int(n_act),), jnp.float32),
    }


def policy_apply(params: ControllerParams, obs: jnp.ndarray) -> jnp.ndarray:
    """obs (..., d_obs) -> activations (..., n_act) in [-1, 1]."""
    h = jnp.tanh(obs @ params["w1"] + params["b1"])
    return jnp.tanh(h @ params["w2"] + params["b2"])


def init_embed(seed: int, n_vox: int, d_e: int) -> jnp.ndarray:
    """E_chi: a fixed (n_vox, d_e) projection of occupancy -> morphology embedding."""
    key = jax.random.PRNGKey(int(seed) + 777)
    return (jax.random.normal(key, (int(n_vox), int(d_e)), jnp.float32)
            / math.sqrt(max(int(n_vox), 1)))


def embed(occ01: jnp.ndarray, E_proj: jnp.ndarray) -> jnp.ndarray:
    """e_x = tanh(occ01 @ E_chi). occ01 (..., n_vox) -> (..., d_e)."""
    return jnp.tanh(occ01 @ E_proj)
