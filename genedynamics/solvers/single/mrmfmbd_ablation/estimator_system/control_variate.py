"""Multi-fidelity control-variate score estimator (task-agnostic, numpy).

Setting (one diffusion step): M proposals θ_1..θ_M. Each has a cheap, biased
low-fidelity reward R^lo_m (all M evaluated) and, for a subset S (|S|=K), an
expensive high-fidelity reward R^hi_m. The MBD score needs the importance-
weighted clean mean  θ̄ = Σ_m softmax(R_m/T) θ_m  with R at HIGH fidelity.

Control-variate (difference) estimator of the high-fidelity reward per proposal:

    R̃_m = R^hi_m                       if m ∈ S
         = R^lo_m + Δ̄                   otherwise,   Δ̄ = mean_{j∈S}(R^hi_j − R^lo_j)

Δ̄ is the control-variate correction: it removes the systematic low-fidelity
bias measured on the subset. Because the per-proposal spread of θ (i.e. of μ_m)
*cancels* in the differences R^hi−R^lo, Var(Δ̄) ≪ Var(mean of K hi-fi draws),
so at equal #high-fidelity evaluations the estimator's variance is far lower —
and at a fixed budget it keeps M large (dense proposal coverage) while staying
hi-fidelity-accurate. As K→M, R̃→R^hi exactly (zero bias, zero CV variance).

This module consumes reward arrays + a subset index only — no simulator / no
soft-robot coupling — so it is reusable on any multi-fidelity reward oracle.
A JAX mirror for the MBD fast-path scan is a thin follow-on (pairs with GPU smoke).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def corrected_rewards(
    R_lo: np.ndarray,
    R_hi_sub: np.ndarray,
    subset_idx: np.ndarray,
    M: int,
) -> np.ndarray:
    """Control-variate corrected high-fidelity reward per proposal (M,).

    Args
    ----
    R_lo      : (M,) low-fidelity reward for every proposal.
    R_hi_sub  : (K,) high-fidelity reward for the subset proposals (aligned to subset_idx).
    subset_idx: (K,) indices in [0, M) that were evaluated at high fidelity.
    """
    R_lo = np.asarray(R_lo, dtype=np.float64)
    R_hi_sub = np.asarray(R_hi_sub, dtype=np.float64)
    subset_idx = np.asarray(subset_idx, dtype=np.int64)
    if subset_idx.size == 0:
        return R_lo.copy()
    delta_bar = float(np.mean(R_hi_sub - R_lo[subset_idx]))
    Rt = R_lo + delta_bar              # debias every proposal by the measured bias
    Rt[subset_idx] = R_hi_sub          # use the true hi-fi reward where we paid for it
    return Rt


def softmax_weights(R: np.ndarray, T: float) -> np.ndarray:
    """Boltzmann weights w_m ∝ exp((R_m − max)/ (std·T)) — matches MBD _weighted_mean."""
    R = np.asarray(R, dtype=np.float64)
    mean = R.mean()
    std = max(float(R.std()), 1e-4)
    logw = (R - mean) / (std * max(float(T), 1e-8))
    logw -= logw.max()
    w = np.exp(logw)
    return w / w.sum()


def weighted_mean(theta: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Importance-weighted clean mean θ̄ = Σ_m w_m θ_m. theta: (M, D) or (M,)."""
    theta = np.asarray(theta, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    return np.tensordot(w, theta, axes=(0, 0))


def cv_score_weighted_mean(
    theta: np.ndarray,
    R_lo: np.ndarray,
    R_hi_sub: np.ndarray,
    subset_idx: np.ndarray,
    T: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full control-variate score: corrected rewards → softmax weights → θ̄.

    Returns (theta_bar, weights, R_tilde).
    """
    M = int(np.asarray(theta).shape[0])
    R_tilde = corrected_rewards(R_lo, R_hi_sub, subset_idx, M)
    w = softmax_weights(R_tilde, T)
    return weighted_mean(theta, w), w, R_tilde


def estimator_diagnostics(
    R_lo: np.ndarray,
    R_hi_sub: np.ndarray,
    subset_idx: np.ndarray,
) -> Dict[str, float]:
    """Correlation + difference-variance diagnostics that justify the CV gain.

    A high lo/hi correlation (or, equivalently, a small variance of the
    differences relative to Var(R_hi)) is exactly when the control variate pays
    off; these are the quantities a budget-allocation rule keys off of.
    """
    R_lo = np.asarray(R_lo, dtype=np.float64)
    R_hi_sub = np.asarray(R_hi_sub, dtype=np.float64)
    subset_idx = np.asarray(subset_idx, dtype=np.int64)
    if subset_idx.size < 2:
        return {"corr": 0.0, "var_diff": 0.0, "var_hi": 0.0, "var_reduction": 0.0}
    lo_s = R_lo[subset_idx]
    diff = R_hi_sub - lo_s
    var_diff = float(np.var(diff))
    var_hi = float(np.var(R_hi_sub))
    if np.std(lo_s) > 1e-12 and np.std(R_hi_sub) > 1e-12:
        corr = float(np.corrcoef(lo_s, R_hi_sub)[0, 1])
    else:
        corr = 0.0
    var_reduction = 1.0 - (var_diff / var_hi) if var_hi > 1e-12 else 0.0
    return {"corr": corr, "var_diff": var_diff, "var_hi": var_hi,
            "var_reduction": var_reduction}
