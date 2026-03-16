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
    p0: float,
    nu0: float = 0.0,
    topK0: int = 8,
    I0: int = 10,
    eps0: float = 1e-4,
    v0: float = 0.0,
    v_ema0: float = 0.0,
    rho_good_count0: int = 0,
) -> Dict[str, jnp.ndarray]:
    """Create initial scheduler carry. All fields are JAX arrays."""
    return {
        "lam": jnp.asarray(lam0, dtype=jnp.float32),
        "rho": jnp.asarray(rho0, dtype=jnp.float32),
        # Compute dual (ergodic compute budget constraint)
        "nu": jnp.asarray(nu0, dtype=jnp.float32),
        # Risk residual history (for λ/ρ updates)
        "rp_prev": jnp.asarray(0.0, dtype=jnp.float32),
        "rp_prev_eff": jnp.asarray(0.0, dtype=jnp.float32),
        # Non-monotone penalty hysteresis counter (consecutive "good" steps)
        "rho_good_count": jnp.asarray(rho_good_count0, dtype=jnp.int32),
        # Controller state: violation signal v_k and its EMA (for non-monotone control)
        "v_prev": jnp.asarray(v0, dtype=jnp.float32),
        "v_ema": jnp.asarray(v_ema0, dtype=jnp.float32),
        # Hysteresis gate for budget penalty (1=budget active, 0=budget inactive)
        "budget_gate": jnp.asarray(0.0, dtype=jnp.float32),
        # Controller state: continuous knob states (rounded/cast in compute_params)
        "p_prev": jnp.asarray(p0, dtype=jnp.float32),
        "topK_state": jnp.asarray(float(topK0), dtype=jnp.float32),
        "I_state": jnp.asarray(float(I0), dtype=jnp.float32),
        "eps_state": jnp.asarray(eps0, dtype=jnp.float32),
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
    # Non-monotone controller config
    v_signal_mode: str,
    v_star: float,
    v_ema_beta: float,
    v_tau_hi: float,
    v_tau_lo: float,
    eta_p: float,
    eta_p_plus: float,
    eta_p_minus: float,
    eta_c_p: float,
    # Non-monotone topK/I/eps controllers
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
    # Compute dual control (ergodic compute budget)
    compute_budget_B: float,
    # Time-varying budget profile (scheme A)
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
    """
    Compute schedule params for current step from carry (no feedback from this step).
    This is a controller-style scheduler: params are produced directly from internal states
    updated in `update()` using violation statistics v_k and compute dual ν_k.
    """
    lam = carry["lam"]
    rho = carry["rho"]
    nu = carry["nu"]
    p_k = carry["p_prev"]
    topK_state = carry["topK_state"]
    I_state = carry["I_state"]
    eps_state = carry["eps_state"]
    rng = carry["rng"]

    # Clip controller states into valid ranges.
    p_k = jnp.clip(p_k, jnp.asarray(p_min, dtype=jnp.float32), jnp.asarray(p_max, dtype=jnp.float32))
    topK_k = jnp.clip(jnp.round(topK_state), topK_min, topK_max).astype(jnp.int32)
    I_k = jnp.clip(jnp.round(I_state), I_min, I_max).astype(jnp.int32)
    eps_k = jnp.clip(eps_state, jnp.asarray(eps_min, dtype=jnp.float32), jnp.asarray(eps_max, dtype=jnp.float32))

    if use_stochastic_gate:
        rng, sub = jax.random.split(rng)
        do_qp = jax.random.uniform(sub) < p_k
    else:
        do_qp = p_k > p_min

    # Compute cost c_k := c(s_k) (no dependence on violation statistics).
    gate = jnp.asarray(do_qp, dtype=jnp.float32) if compute_cost_use_qp_gate else jnp.asarray(1.0, dtype=jnp.float32)
    if compute_cost_mode == "product":
        # c_k = gate * p_k * topK_k * I_k
        compute_cost_hat = gate * p_k * topK_k.astype(jnp.float32) * I_k.astype(jnp.float32)
    else:
        # Default: c_k = gate * p_k * (a0 + aK*topK_k + aI*I_k)
        cost_shape = (
            jnp.asarray(compute_cost_a0, dtype=jnp.float32)
            + jnp.asarray(compute_cost_aK, dtype=jnp.float32) * topK_k.astype(jnp.float32)
            + jnp.asarray(compute_cost_aI, dtype=jnp.float32) * I_k.astype(jnp.float32)
        )
        compute_cost_hat = gate * p_k * cost_shape

    # Margin: use base (could add schedule later)
    margin = jnp.asarray(margin_base, dtype=jnp.float32)

    # Time-varying compute budget B_k (scheme A): low-high-low across diffusion.
    # step_k is scan progress: 0 = most noisy/early, K-1 = clean/late.
    B_base = jnp.asarray(compute_budget_B, dtype=jnp.float32)
    if compute_budget_time_profile == "gaussian_bump":
        denom = jnp.maximum(jnp.asarray(float(K - 1), dtype=jnp.float32), 1.0)
        t = step_k.astype(jnp.float32) / denom  # 0..1
        a = jnp.maximum(jnp.asarray(compute_budget_time_amp, dtype=jnp.float32), 0.0)
        mu = jnp.asarray(compute_budget_time_mu, dtype=jnp.float32)
        sigma = jnp.maximum(jnp.asarray(compute_budget_time_sigma, dtype=jnp.float32), 1e-3)

        z = (t - mu) / sigma
        w_k = 1.0 + a * jnp.exp(-0.5 * z * z)

        # Normalize so that mean_k B_k = B (ergodic budget preserved).
        # K is a Python int (static), so XLA can treat this as compile-time constant.
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
        # Bookkeeping (used for logging + controller feedback)
        "nu": nu,
        "compute_cost_hat": compute_cost_hat,
        "compute_budget_B": B_base,
        "compute_budget_B_eff": B_eff,
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
    rho_min: float,
    # Non-monotone ALM (λ forgetting + ρ hysteresis)
    eta_lam_forget: float,
    rho_tau_hi: float,
    rho_tau_lo: float,
    rho_gamma_up: float,
    rho_gamma_down: float,
    rho_good_steps: int,
    p_min: float,
    p_max: float,
    eps_min: float,
    eps_max: float,
    I_min: int,
    I_max: int,
    topK_min: int,
    topK_max: int,
    # Non-monotone controller config
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
    # Compute dual control (ergodic compute budget)
    eta_nu: float,
    compute_budget_B: float,
    nu_max: float,
) -> Dict[str, jnp.ndarray]:
    """
    Update carry from feedback.

    Notation:
    - r_p: risk residual r_k (quantile/CVaR of per-trajectory violation magnitudes)
    - v_k: violation statistic used for control (rate or mean), from feedback
    - c_k: compute cost, from feedback
    - nu: compute dual

    Use r_p_eff = max(0, r_p - r_tol) for λ/ρ.
    for λ/ρ; stall only when r_p_eff > r_tol and r_p_eff > kappa * rp_prev_eff.
    """
    lam = carry["lam"]
    rho = carry["rho"]
    nu = carry["nu"]
    rp_prev_eff = carry["rp_prev_eff"]
    rho_good_count = carry["rho_good_count"]
    v_prev = carry["v_prev"]
    v_ema = carry["v_ema"]
    budget_gate_prev = carry["budget_gate"]
    p_prev = carry["p_prev"]
    topK_state = carry["topK_state"]
    I_state = carry["I_state"]
    eps_state = carry["eps_state"]

    r_p = feedback["r_p"]
    compute_cost_hat = feedback["compute_cost_hat"]
    compute_budget_B_eff = feedback["compute_budget_B_eff"]
    v_k_rate = feedback["v_k_rate"]
    v_k_mean = feedback["v_k_mean"]

    r_p_eff = jnp.maximum(0.0, r_p - r_tol)

    # ---------------------------------------------------------------------
    # Non-monotone ALM updates (stochastic / inexact setting):
    #
    # λ update (proximal / regularized dual ascent):
    #   λ_{k+1} = Π_+( (1-η) λ_k + ρ_k r_k )
    # This allows λ to decrease and keeps it bounded (dual regularization).
    #
    # ρ update (hysteresis / non-monotone continuation):
    #   if r_k > τ_hi: ρ ← min(γ_up ρ, ρ_max)
    #   if r_k < τ_lo for W steps: ρ ← max(ρ/γ_down, ρ_min)
    # ---------------------------------------------------------------------
    tau_hi = jnp.asarray(rho_tau_hi, dtype=jnp.float32)
    tau_lo = jnp.asarray(rho_tau_lo, dtype=jnp.float32)
    rho_min_f = jnp.asarray(rho_min, dtype=jnp.float32)
    rho_max_f = jnp.asarray(rho_max, dtype=jnp.float32)
    gamma_up = jnp.asarray(rho_gamma_up, dtype=jnp.float32)
    gamma_down = jnp.asarray(rho_gamma_down, dtype=jnp.float32)
    W = jnp.asarray(max(1, int(rho_good_steps)), dtype=jnp.int32)

    # Update "good" counter: counts consecutive steps with small residual
    rho_good_count_next = jnp.where(
        r_p_eff < tau_lo,
        jnp.minimum(rho_good_count + 1, W),
        jnp.asarray(0, dtype=jnp.int32),
    )

    # Increase rho immediately if violation is large
    rho_inc = jnp.minimum(gamma_up * rho, rho_max_f)
    # Decrease rho only after W consecutive good steps
    rho_dec = jnp.maximum(rho / gamma_down, rho_min_f)
    rho_next = jnp.where(
        r_p_eff > tau_hi,
        rho_inc,
        jnp.where(rho_good_count_next >= W, rho_dec, rho),
    )

    # Reset good counter if we take a decrease or an increase
    rho_good_count_next = jnp.where(
        (r_p_eff > tau_hi) | (rho_good_count_next >= W),
        jnp.asarray(0, dtype=jnp.int32),
        rho_good_count_next,
    )

    # λ update with forgetting (use rho_next for current penalty scale)
    eta_lam = jnp.clip(jnp.asarray(eta_lam_forget, dtype=jnp.float32), 0.0, 1.0)
    lam_next = jnp.maximum(0.0, (1.0 - eta_lam) * lam + rho_next * r_p_eff)

    # Choose violation signal v_k for control (rate or mean). Use EMA to reduce noise.
    v_k = v_k_rate if v_signal_mode == "rate" else v_k_mean
    beta = jnp.asarray(v_ema_beta, dtype=jnp.float32)
    v_ema_next = (1.0 - beta) * v_ema + beta * jnp.asarray(v_k, dtype=jnp.float32)
    dv = v_ema_next - v_prev
    e = v_ema_next - jnp.asarray(v_star, dtype=jnp.float32)

    # Priority rule: budget must NEVER prevent responding to violation.
    #
    # Less conservative gating than "EMA only":
    # - Turn OFF budget penalty if *either* signal indicates high violation.
    # - Turn ON budget penalty if *either* signal indicates sufficiently low violation
    #   (this lets us save compute quickly once v_k drops, even if EMA lags).
    #
    # Hysteresis:
    # - if v_hi > tau_hi: disable budget penalty
    # - if v_lo < tau_lo: enable budget penalty
    # - otherwise: keep previous gate
    tau_hi = jnp.asarray(v_tau_hi, dtype=jnp.float32)
    tau_lo = jnp.asarray(v_tau_lo, dtype=jnp.float32)
    v_k_f = jnp.asarray(v_k, dtype=jnp.float32)
    v_hi = jnp.maximum(v_ema_next, v_k_f)
    v_lo = jnp.minimum(v_ema_next, v_k_f)
    budget_gate = jnp.where(
        v_hi > tau_hi,
        0.0,
        jnp.where(v_lo < tau_lo, 1.0, budget_gate_prev),
    ).astype(jnp.float32)

    # Non-monotone controller updates.
    relu = lambda x: jnp.maximum(0.0, x)
    rebound = jnp.asarray(eta_p_plus, dtype=jnp.float32) * relu(dv) - jnp.asarray(eta_p_minus, dtype=jnp.float32) * relu(-dv)
    p_next = p_prev + jnp.asarray(eta_p, dtype=jnp.float32) * e + rebound - budget_gate * (jnp.asarray(eta_c_p, dtype=jnp.float32) * nu)
    p_next = jnp.clip(p_next, jnp.asarray(p_min, dtype=jnp.float32), jnp.asarray(p_max, dtype=jnp.float32))

    topK_next = topK_state
    topK_next = topK_next + jnp.asarray(eta_topK, dtype=jnp.float32) * e
    topK_next = topK_next + jnp.asarray(eta_topK_plus, dtype=jnp.float32) * relu(dv) - jnp.asarray(eta_topK_minus, dtype=jnp.float32) * relu(-dv)
    topK_next = topK_next - budget_gate * (jnp.asarray(eta_c_topK, dtype=jnp.float32) * nu)
    topK_next = jnp.clip(topK_next, jnp.asarray(float(topK_min), dtype=jnp.float32), jnp.asarray(float(topK_max), dtype=jnp.float32))

    I_next = I_state
    I_next = I_next + jnp.asarray(eta_I, dtype=jnp.float32) * e
    I_next = I_next + jnp.asarray(eta_I_plus, dtype=jnp.float32) * relu(dv) - jnp.asarray(eta_I_minus, dtype=jnp.float32) * relu(-dv)
    I_next = I_next - budget_gate * (jnp.asarray(eta_c_I, dtype=jnp.float32) * nu)
    I_next = jnp.clip(I_next, jnp.asarray(float(I_min), dtype=jnp.float32), jnp.asarray(float(I_max), dtype=jnp.float32))

    # eps: smaller is more accurate (more compute). Decrease when violation increases; increase when it decreases.
    eps_next = eps_state
    eps_next = eps_next - jnp.asarray(eta_eps, dtype=jnp.float32) * e
    eps_next = eps_next - jnp.asarray(eta_eps_plus, dtype=jnp.float32) * relu(dv) + jnp.asarray(eta_eps_minus, dtype=jnp.float32) * relu(-dv)
    eps_next = eps_next + budget_gate * (jnp.asarray(eta_c_eps, dtype=jnp.float32) * nu)
    eps_next = jnp.clip(eps_next, jnp.asarray(eps_min, dtype=jnp.float32), jnp.asarray(eps_max, dtype=jnp.float32))

    # Compute dual update (projected gradient ascent on dual):
    # nu_{k+1} = [nu_k + eta_nu * (c_hat(s_k) - B)]_+
    B_eff = jnp.asarray(compute_budget_B_eff, dtype=jnp.float32)
    nu_step = jnp.asarray(eta_nu, dtype=jnp.float32) * (compute_cost_hat - B_eff)
    nu_next = jnp.maximum(0.0, nu + nu_step)
    if float(nu_max) > 0:
        nu_next = jnp.minimum(nu_next, jnp.asarray(nu_max, dtype=jnp.float32))

    return {
        "lam": lam_next,
        "rho": rho_next,
        "nu": nu_next,
        "rp_prev": r_p,
        "rp_prev_eff": r_p_eff,
        "rho_good_count": rho_good_count_next,
        "p_prev": p_next,
        "topK_state": topK_next,
        "I_state": I_next,
        "eps_state": eps_next,
        "v_prev": v_ema_next,
        "v_ema": v_ema_next,
        "budget_gate": budget_gate,
        "rng": rng,
    }
