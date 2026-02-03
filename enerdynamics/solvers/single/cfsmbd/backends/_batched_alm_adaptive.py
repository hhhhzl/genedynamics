"""
Batched JAX backend for ALMAdaptiveConstraintScheduler.

This module provides drop-in replacements for:
- `enerdynamics.core.constraints.schedulers.ConstraintScheduler.almadaptive.backends.alm_adaptive_jax.init_carry`
- `enerdynamics.core.constraints.schedulers.ConstraintScheduler.almadaptive.backends.alm_adaptive_jax.compute_params`

Key features:
- Supports `rng` shaped (2,) OR (C,2) (or any leading batch dims) without needing `vmap`.
- Returns carry/params where every field is a JAX array with matching leading shape.

We use runtime monkey-patching from the CFS-MBD JAX backend to avoid editing the
core scheduler files (which are not writable in this environment).
"""

from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp


def _split_key_batched(key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Split PRNGKey with optional leading batch dims."""
    key = jnp.asarray(key)
    if key.ndim == 1:
        k1, k2 = jax.random.split(key, 2)
        return k1, k2
    lead = key.shape[:-1]
    key_flat = key.reshape((-1, 2))
    keys_flat = jax.vmap(lambda k: jax.random.split(k, 2), in_axes=0)(key_flat)  # (B,2,2)
    k1 = keys_flat[:, 0, :].reshape(lead + (2,))
    k2 = keys_flat[:, 1, :].reshape(lead + (2,))
    return k1, k2


def _uniform01_batched(key: jnp.ndarray) -> jnp.ndarray:
    """Uniform[0,1) with optional leading batch dims on key."""
    key = jnp.asarray(key)
    if key.ndim == 1:
        return jax.random.uniform(key)
    lead = key.shape[:-1]
    key_flat = key.reshape((-1, 2))
    u_flat = jax.vmap(jax.random.uniform, in_axes=0)(key_flat)  # (B,)
    return u_flat.reshape(lead)


def init_carry(
    rng: jnp.ndarray,
    lam0: float,
    rho0: float,
    p0: float,
    nu0: float = 0.0,
    topK0: int = 8,
    I0: int = 10,
    eps0: float = 1e-4,
    v0: float = 0.0,
    v_ema0: float = 0.0,
    rho_good_count0: int = 0,
) -> Dict[str, jnp.ndarray]:
    """Batched-compatible initial carry (all fields are arrays)."""
    rng = jnp.asarray(rng)
    lead = tuple(rng.shape[:-1])  # () for scalar key, (C,) for (C,2)
    ones_f = jnp.ones(lead, dtype=jnp.float32)
    ones_i = jnp.ones(lead, dtype=jnp.int32)
    return {
        "lam": jnp.asarray(lam0, dtype=jnp.float32) * ones_f,
        "rho": jnp.asarray(rho0, dtype=jnp.float32) * ones_f,
        "nu": jnp.asarray(nu0, dtype=jnp.float32) * ones_f,
        "rp_prev": jnp.asarray(0.0, dtype=jnp.float32) * ones_f,
        "rp_prev_eff": jnp.asarray(0.0, dtype=jnp.float32) * ones_f,
        "rho_good_count": jnp.asarray(rho_good_count0, dtype=jnp.int32) * ones_i,
        "v_prev": jnp.asarray(v0, dtype=jnp.float32) * ones_f,
        "v_ema": jnp.asarray(v_ema0, dtype=jnp.float32) * ones_f,
        "budget_gate": jnp.asarray(0.0, dtype=jnp.float32) * ones_f,
        "p_prev": jnp.asarray(p0, dtype=jnp.float32) * ones_f,
        "topK_state": jnp.asarray(float(topK0), dtype=jnp.float32) * ones_f,
        "I_state": jnp.asarray(float(I0), dtype=jnp.float32) * ones_f,
        "eps_state": jnp.asarray(eps0, dtype=jnp.float32) * ones_f,
        "rng": rng,
    }


def compute_params(
    carry: Dict[str, jnp.ndarray],
    step_k: jnp.ndarray,
    K: int,
    margin_base: float,
    gamma: float,
    rho_max: float,
    kappa: float,
    p_min: float,
    p_max: float,
    eps_min: float,
    eps_max: float,
    I_min: int,
    I_max: int,
    topK_min: int,
    topK_max: int,
    use_stochastic_gate: bool,
    v_signal_mode: str,
    v_star: float,
    v_ema_beta: float,
    v_tau_hi: float,
    v_tau_lo: float,
    eta_p: float,
    eta_p_plus: float,
    eta_p_minus: float,
    eta_c_p: float,
    eta_topK: float,
    eta_topK_plus: float,
    eta_topK_minus: float,
    eta_c_topK: float,
    eta_I: float,
    eta_I_plus: float,
    eta_I_minus: float,
    eta_c_I: float,
    eta_eps: float,
    eta_eps_plus: float,
    eta_eps_minus: float,
    eta_c_eps: float,
    compute_budget_B: float,
    compute_budget_time_profile: str,
    compute_budget_time_amp: float,
    compute_budget_time_mu: float,
    compute_budget_time_sigma: float,
    compute_cost_mode: str,
    compute_cost_a0: float,
    compute_cost_aK: float,
    compute_cost_aI: float,
    compute_cost_use_qp_gate: bool,
):
    """Batched-compatible compute_params (drop-in replacement for core backend)."""
    lam = carry["lam"]
    rho = carry["rho"]
    nu = carry["nu"]
    p_k = carry["p_prev"]
    topK_state = carry["topK_state"]
    I_state = carry["I_state"]
    eps_state = carry["eps_state"]
    rng = carry["rng"]

    p_k = jnp.clip(p_k, jnp.asarray(p_min, dtype=jnp.float32), jnp.asarray(p_max, dtype=jnp.float32))
    topK_k = jnp.clip(jnp.round(topK_state), topK_min, topK_max).astype(jnp.int32)
    I_k = jnp.clip(jnp.round(I_state), I_min, I_max).astype(jnp.int32)
    eps_k = jnp.clip(eps_state, jnp.asarray(eps_min, dtype=jnp.float32), jnp.asarray(eps_max, dtype=jnp.float32))

    # Leading batch shape ((), (C,), or higher-rank) inferred from controller state.
    # Use this to broadcast scalar config values (e.g., margin) to per-mode arrays.
    ones_f = jnp.ones_like(p_k, dtype=jnp.float32)

    if use_stochastic_gate:
        rng_next, sub = _split_key_batched(rng)
        u = _uniform01_batched(sub)
        do_qp = u < p_k
        rng_out = rng_next
    else:
        do_qp = p_k > p_min
        rng_out = rng

    gate = jnp.asarray(do_qp, dtype=jnp.float32) if compute_cost_use_qp_gate else jnp.asarray(1.0, dtype=jnp.float32)
    if compute_cost_mode == "product":
        compute_cost_hat = gate * p_k * topK_k.astype(jnp.float32) * I_k.astype(jnp.float32)
    else:
        cost_shape = (
            jnp.asarray(compute_cost_a0, dtype=jnp.float32)
            + jnp.asarray(compute_cost_aK, dtype=jnp.float32) * topK_k.astype(jnp.float32)
            + jnp.asarray(compute_cost_aI, dtype=jnp.float32) * I_k.astype(jnp.float32)
        )
        compute_cost_hat = gate * p_k * cost_shape

    margin = jnp.asarray(margin_base, dtype=jnp.float32) * ones_f

    B_base = jnp.asarray(compute_budget_B, dtype=jnp.float32) * ones_f
    if compute_budget_time_profile == "gaussian_bump":
        denom = jnp.maximum(jnp.asarray(float(K - 1), dtype=jnp.float32), 1.0)
        t = step_k.astype(jnp.float32) / denom
        a = jnp.maximum(jnp.asarray(compute_budget_time_amp, dtype=jnp.float32), 0.0)
        mu = jnp.asarray(compute_budget_time_mu, dtype=jnp.float32)
        sigma = jnp.maximum(jnp.asarray(compute_budget_time_sigma, dtype=jnp.float32), 1e-3)
        z = (t - mu) / sigma
        w_k = 1.0 + a * jnp.exp(-0.5 * z * z)
        t_grid = jnp.linspace(0.0, 1.0, K, dtype=jnp.float32)
        z_grid = (t_grid - mu) / sigma
        w_grid = 1.0 + a * jnp.exp(-0.5 * z_grid * z_grid)
        mean_w = jnp.mean(w_grid)
        B_eff = B_base * (w_k / mean_w)
    else:
        B_eff = B_base

    params = {
        "margin": margin,
        "rho": rho,
        "qp_gate": do_qp,
        "qp_prob": p_k,
        "topK": topK_k,
        "eps": eps_k,
        "I_QP": I_k,
        "aug_lambda": lam,
        "aug_rho": rho,
        "nu": nu,
        "compute_cost_hat": compute_cost_hat,
        "compute_budget_B": B_base,
        "compute_budget_B_eff": B_eff,
    }
    return params, rng_out

