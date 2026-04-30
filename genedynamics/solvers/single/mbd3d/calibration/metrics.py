"""
Calibration metrics for per-pixel posterior over rendered RGB.

Given:
  - `gt`  (N, H, W, 3) — ground-truth held-out images in [0, 1]
  - `mean` (N, H, W, 3) — posterior mean prediction
  - `std`  (N, H, W, 3) — posterior std (channel-wise)

we compute reliability, ECE, coverage, and NLL. All metrics are aggregated over
the flattened sample set (N * H * W * 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CalibrationResult:
    bin_edges: np.ndarray           # (B+1,)
    bin_pred_std: np.ndarray        # (B,) mean predicted std in each bin
    bin_actual_err: np.ndarray      # (B,) mean absolute err in each bin
    bin_counts: np.ndarray          # (B,)
    ece: float                      # scalar


def _flatten(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    return tuple(np.asarray(a, dtype=np.float32).reshape(-1) for a in arrays)


def compute_reliability_curve(
    gt: np.ndarray, mean: np.ndarray, std: np.ndarray, *, n_bins: int = 15
) -> CalibrationResult:
    """
    Bin pixels by predicted std and compare to observed absolute residual.

    A perfectly-calibrated Gaussian posterior satisfies E[|x - μ|] ≈ σ · sqrt(2/π).
    We report the scaled residual (|err| / sqrt(2/π)) so that bins on the
    identity line indicate perfect calibration.
    """
    gt_f, mu_f, sd_f = _flatten(gt, mean, std)
    err = np.abs(gt_f - mu_f) / np.sqrt(2.0 / np.pi)

    sd_min, sd_max = float(sd_f.min()), float(sd_f.max())
    if sd_max <= sd_min + 1e-12:
        sd_max = sd_min + 1e-6
    edges = np.linspace(sd_min, sd_max, n_bins + 1)
    bin_idx = np.clip(np.digitize(sd_f, edges[1:-1]), 0, n_bins - 1)

    pred_per_bin = np.zeros(n_bins, dtype=np.float32)
    actual_per_bin = np.zeros(n_bins, dtype=np.float32)
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = bin_idx == b
        counts[b] = int(mask.sum())
        if counts[b] > 0:
            pred_per_bin[b] = float(sd_f[mask].mean())
            actual_per_bin[b] = float(err[mask].mean())

    nonzero = counts > 0
    if nonzero.any():
        w = counts[nonzero].astype(np.float32) / counts.sum()
        ece = float(np.sum(w * np.abs(pred_per_bin[nonzero] - actual_per_bin[nonzero])))
    else:
        ece = 0.0
    return CalibrationResult(
        bin_edges=edges,
        bin_pred_std=pred_per_bin,
        bin_actual_err=actual_per_bin,
        bin_counts=counts,
        ece=ece,
    )


def compute_ece(gt: np.ndarray, mean: np.ndarray, std: np.ndarray, *, n_bins: int = 15) -> float:
    return compute_reliability_curve(gt, mean, std, n_bins=n_bins).ece


def compute_coverage_curve(
    gt: np.ndarray, mean: np.ndarray, std: np.ndarray,
    *, levels: Tuple[float, ...] = (0.5, 0.68, 0.90, 0.95, 0.99),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Empirical coverage at requested credible-interval levels.

    Returns (levels, coverages) — `coverages[i]` is the fraction of pixels
    whose residual |x - μ| ≤ z_i · σ, where z_i is the one-sided Gaussian
    quantile for the two-sided probability `levels[i]`.
    """
    from scipy.stats import norm

    gt_f, mu_f, sd_f = _flatten(gt, mean, std)
    sd_f = np.maximum(sd_f, 1e-8)
    z = np.asarray([norm.ppf(0.5 + 0.5 * L) for L in levels], dtype=np.float32)
    abs_resid = np.abs(gt_f - mu_f) / sd_f
    coverages = np.asarray(
        [float((abs_resid <= zi).mean()) for zi in z], dtype=np.float32
    )
    return np.asarray(levels, dtype=np.float32), coverages


def compute_pixel_nll(
    gt: np.ndarray, mean: np.ndarray, std: np.ndarray, *, eps: float = 1e-6,
) -> float:
    """Per-pixel Gaussian negative log-likelihood, averaged over all pixels."""
    gt_f, mu_f, sd_f = _flatten(gt, mean, std)
    sd_f = np.maximum(sd_f, eps)
    ll = -0.5 * np.log(2 * np.pi * sd_f ** 2) - 0.5 * ((gt_f - mu_f) / sd_f) ** 2
    return float(-ll.mean())
