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


REGIME_POSTERIOR_MODES = ("reward", "risk_sensitive")


def validate_mode(mode: str) -> str:
    """Raise unless `mode` is a known regime_posterior_mode value."""
    if mode not in REGIME_POSTERIOR_MODES:
        raise ValueError(
            f"Unknown regime_posterior_mode: {mode!r}. "
            f"Allowed: {REGIME_POSTERIOR_MODES}."
        )
    return mode
