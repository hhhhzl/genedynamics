"""Risk-sensitive regime posterior — writeup §5 / §6 implementation.

The existing `_s1_marginalize_jax` in marginalizer.py implements REWARD
marginalization:

    R_m       = logsumexp_c (log p(c) + R_{m,c} / T)
    q(c | m)  = softmax_c   (log p(c) + R_{m,c} / T)

This pulls the posterior toward HIGH-reward regimes — i.e. easy regimes
dominate. The writeup wants the OPPOSITE for robustness: weight up
FAILURE-PRONE (low-reward) regimes so the optimizer can't game the average
by overfitting to flat ground.

The risk-sensitive marginalizer uses the soft-min formulation:

    rho(z)    = -tau_r * log Σ_m p(m) exp(-R(z; m) / tau_r)
    q(m | z)  = softmax_m (log p(m) - R(z; m) / tau_r)

Limits
------
- tau_r → 0     : worst-regime objective (max-min)
- tau_r → ∞     : uniform expectation = arithmetic mean over regimes
- tau_r ≈ R_std : balanced — common practical setting

Sign convention matches the MBD backend's `_s1_marginalize_jax` return shape
so the two functions are drop-in interchangeable: both return
`(per_candidate_score (M,), per_(M,C)_responsibilities)`.
"""

from __future__ import annotations

from typing import Tuple

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def risk_sensitive_marginalize_jax(
    rewards_mc: "jnp.ndarray",   # (M, C) — M candidates × C regimes
    log_prior: "jnp.ndarray",    # (C,)
    tau_r: "jnp.ndarray",        # scalar — risk temperature
) -> Tuple["jnp.ndarray", "jnp.ndarray"]:
    """Risk-sensitive (soft-min) regime marginalization.

    Returns
    -------
    rho_m : (M,)
        Per-candidate risk-sensitive score; high = robust across regimes.
        Same sign convention as `_s1_marginalize_jax`'s return (higher is
        better), so the MBD reward-weighted-mean math is unchanged.
    q_mc : (M, C)
        Per-candidate posterior over regimes — emphasizes failure-prone
        regimes (the writeup §5 q(m|z) for adaptive computation allocation).
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is required for risk_sensitive_marginalize_jax")
    rewards_mc = jnp.nan_to_num(rewards_mc, nan=-1e6, posinf=1e6, neginf=-1e6)
    tau_eff = jnp.maximum(tau_r, jnp.asarray(1e-8, dtype=rewards_mc.dtype))
    # log p(m) - R_{m,c} / tau_r  → high when reward LOW (failure-prone).
    log_terms = log_prior[None, :] - rewards_mc / tau_eff
    # Soft-min over regimes: rho_m = -tau_r * lse_c(log p - R/tau_r).
    lse = jax.scipy.special.logsumexp(log_terms, axis=-1)
    rho_m = -tau_eff * lse
    q_mc = jnp.exp(jax.nn.log_softmax(log_terms, axis=-1))
    return rho_m, q_mc


def cvar_marginalize_jax(
    rewards_mc: "jnp.ndarray",   # (M, C) — M candidates × C regimes
    log_prior: "jnp.ndarray",    # (C,)
    alpha: "jnp.ndarray",        # scalar in (0, 1] — CVaR tail level
) -> Tuple["jnp.ndarray", "jnp.ndarray"]:
    """CVaR_α regime marginalization (distributionally-robust / adversarial).

    For each candidate, ``rho_m`` is the average reward over the WORST
    α-probability mass of regimes (the lower tail), and ``q_mc`` is the
    adversarial regime posterior CVaR induces — all mass on the failing
    α-tail, p(m)/α each.

    A different robustness knob from `risk_sensitive_marginalize_jax`: that one
    is the KL-ball (entropic) DRO posterior q ∝ p·exp(-R/τ_r); CVaR is the
    {q ≤ p/α} ambiguity set, a hard tail average with a crisp "worst α-fraction"
    reading. Computed by the exact discrete formula: sort regimes ascending,
    take the probability mass that overlaps the worst-α quantile.

    Limits
    ------
    - α → 0 : worst-regime objective (max-min), q → one-hot worst regime
    - α = 1 : prior-mean over regimes, q → p(m)

    Same return signature/sign as the other marginalizers (higher rho = more
    robust), so it is drop-in interchangeable in the MBD weighting math.
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is required for cvar_marginalize_jax")
    rewards_mc = jnp.nan_to_num(rewards_mc, nan=-1e6, posinf=1e6, neginf=-1e6)
    p = jnp.exp(log_prior)
    p = p / jnp.maximum(jnp.sum(p), jnp.asarray(1e-12, dtype=p.dtype))
    a = jnp.clip(alpha, jnp.asarray(1e-6, dtype=rewards_mc.dtype), jnp.asarray(1.0, dtype=rewards_mc.dtype))

    def _row(R):
        order = jnp.argsort(R)                 # ascending: worst regimes first
        R_s = R[order]
        p_s = p[order]
        cum = jnp.cumsum(p_s)
        cum_before = cum - p_s
        # Probability mass of each regime that falls inside the worst-α quantile.
        overlap = jnp.clip(jnp.minimum(cum, a) - cum_before, 0.0, None)
        w_s = overlap / a                      # adversarial tail weights, Σ = 1
        cvar = jnp.sum(w_s * R_s)
        q = jnp.zeros_like(R).at[order].set(w_s)
        return cvar, q

    rho_m, q_mc = jax.vmap(_row)(rewards_mc)
    return rho_m, q_mc


REGIME_POSTERIOR_MODES = ("reward", "risk_sensitive", "cvar")


def validate_mode(mode: str) -> str:
    """Raise unless `mode` is a known regime_posterior_mode value."""
    if mode not in REGIME_POSTERIOR_MODES:
        raise ValueError(
            f"Unknown regime_posterior_mode: {mode!r}. "
            f"Allowed: {REGIME_POSTERIOR_MODES}."
        )
    return mode
