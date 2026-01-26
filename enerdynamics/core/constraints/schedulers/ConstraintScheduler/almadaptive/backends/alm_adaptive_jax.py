"""
JAX backend for ALM adaptive constraint scheduler.

Pure JAX functions for use inside lax.scan: init_carry, compute_params, update.
All ALM and QP-effort logic lives here; CFS-MBD only calls these and uses returned params.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp


def quantile_90(v: jnp.ndarray, k_top: int) -> jnp.ndarray:
    """Approximate 90th percentile via top-k. JIT-safe. k_top = top 10% size (Python int)."""
    k = min(max(1, k_top), v.shape[0])
    top_vals, _ = jax.lax.top_k(v, k)
    return jnp.min(top_vals)


def init_carry(
    rng: jnp.ndarray,
    lam0: float,
    rho0: float,
    p_max: float,
) -> Dict[str, jnp.ndarray]:
    """Create initial scheduler carry. All fields are JAX arrays."""
    return {
        "lam": jnp.asarray(lam0, dtype=jnp.float32),
        "rho": jnp.asarray(rho0, dtype=jnp.float32),
        "rp_prev": jnp.asarray(0.0, dtype=jnp.float32),
        "rp_prev_eff": jnp.asarray(0.0, dtype=jnp.float32),
        "proj_prev": jnp.asarray(0.0, dtype=jnp.float32),
        "p_prev": jnp.asarray(p_max, dtype=jnp.float32),
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
    r_scale: float,
    proj_min: float,
    alpha_smooth: float,
    use_stochastic_gate: bool,
):
    """
    Compute schedule params for current step from carry (no feedback from this step).
    Uses rp_prev, proj_prev for effort mapping.
    """
    lam = carry["lam"]
    rho = carry["rho"]
    rp_prev = carry["rp_prev"]
    proj_prev = carry["proj_prev"]
    p_prev = carry["p_prev"]
    rng = carry["rng"]

    # Effort: p_k from rp_prev (higher violation -> more QP), reduced if proj_prev small
    p_raw = p_min + (p_max - p_min) * jnp.tanh(rp_prev / (r_scale + 1e-9))
    reduce = jnp.where(proj_prev < proj_min, 0.2, 0.0)
    p_new = jnp.clip(p_raw - reduce, p_min, p_max)
    p_k = (1.0 - alpha_smooth) * p_prev + alpha_smooth * p_new

    # eps, I_QP, topK scale with p_k
    eps_k = eps_max * (1.0 - p_k) + eps_min * p_k
    I_k = I_min + jnp.round((I_max - I_min) * p_k).astype(jnp.int32)
    topK_k = topK_min + jnp.round((topK_max - topK_min) * p_k).astype(jnp.int32)

    if use_stochastic_gate:
        rng, sub = jax.random.split(rng)
        do_qp = jax.random.uniform(sub) < p_k
    else:
        do_qp = p_k > p_min

    # Margin: use base (could add schedule later)
    margin = jnp.asarray(margin_base, dtype=jnp.float32)

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
    }
    return params, rng  # rng is updated after split; pass to update for new_carry


def update(
    carry: Dict[str, jnp.ndarray],
    feedback: Dict[str, jnp.ndarray],
    rng: jnp.ndarray,
    gamma: float,
    rho_max: float,
    kappa: float,
    r_tol: float,
    p_min: float,
    p_max: float,
    alpha_smooth: float,
    r_scale: float,
    proj_min: float,
) -> Dict[str, jnp.ndarray]:
    """
    Update carry from feedback (r_p, proj). Use r_p_eff = max(0, r_p - r_tol)
    for λ/ρ; stall only when r_p_eff > r_tol and r_p_eff > kappa * rp_prev_eff.
    """
    lam = carry["lam"]
    rho = carry["rho"]
    rp_prev = carry["rp_prev"]
    rp_prev_eff = carry["rp_prev_eff"]
    p_prev = carry["p_prev"]

    r_p = feedback["r_p"]
    proj = feedback["proj"]

    r_p_eff = jnp.maximum(0.0, r_p - r_tol)

    # Dual update (use effective residual)
    lam_next = jnp.maximum(0.0, lam + rho * r_p_eff)

    # Penalty: grow only on stall — significant residual AND no progress
    stall = (r_p_eff > r_tol) & (r_p_eff > kappa * (rp_prev_eff + 1e-12))
    rho_next = jnp.where(
        stall,
        jnp.minimum(gamma * rho, jnp.asarray(rho_max, dtype=jnp.float32)),
        rho,
    )

    # Effort smoothing: p_prev for next step (still use raw r_p for effort)
    p_raw = p_min + (p_max - p_min) * jnp.tanh(r_p / (r_scale + 1e-9))
    reduce = jnp.where(proj < proj_min, 0.2, 0.0)
    p_new = jnp.clip(p_raw - reduce, p_min, p_max)
    p_next = (1.0 - alpha_smooth) * p_prev + alpha_smooth * p_new

    return {
        "lam": lam_next,
        "rho": rho_next,
        "rp_prev": r_p,
        "rp_prev_eff": r_p_eff,
        "proj_prev": proj,
        "p_prev": p_next,
        "rng": rng,
    }
