#!/usr/bin/env python3
"""
Plot a posterior-calibration reliability curve from saved render artifacts.

Expects a result directory with:
  - gt_images.npy     (N, H, W, 3)
  - mean_pred.npy     (N, H, W, 3)
  - pred_stack.npy    (M, N, H, W, 3)     # posterior samples, ≥ 2 seeds
  - var_pred.npy      (N, H, W) or (N, H, W, 3)

Usage:
  python scripts/tasks/3dgs/plot_calibration_curve.py <result_dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from genedynamics.solvers.single.mbd3d.calibration import (
    compute_reliability_curve, compute_coverage_curve, compute_pixel_nll,
)


def _infer_std(var: np.ndarray, mean: np.ndarray) -> np.ndarray:
    if var.ndim == 3:
        # (N, H, W) → broadcast to (N, H, W, 3)
        var = np.repeat(var[..., None], mean.shape[-1], axis=-1)
    return np.sqrt(np.maximum(var, 1e-12))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=str)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--output", type=str, default=None,
                        help="PNG output path (defaults to <result_dir>/calibration_curve.png)")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    gt = np.load(result_dir / "gt_images.npy")
    mean = np.load(result_dir / "mean_pred.npy")
    var = np.load(result_dir / "var_pred.npy")
    std = _infer_std(var, mean)

    reliability = compute_reliability_curve(gt, mean, std, n_bins=args.n_bins)
    levels, coverages = compute_coverage_curve(gt, mean, std)
    nll = compute_pixel_nll(gt, mean, std)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Reliability
    ax = axes[0]
    mask = reliability.bin_counts > 0
    x = reliability.bin_pred_std[mask]
    y = reliability.bin_actual_err[mask]
    lim = float(max(x.max() if x.size else 1e-3, y.max() if y.size else 1e-3))
    ax.plot([0, lim], [0, lim], ":", color="gray", label="perfectly calibrated")
    ax.plot(x, y, "o-", color="#1f77b4", markersize=6, linewidth=1.8)
    ax.set_xlabel("Predicted std (per-pixel)")
    ax.set_ylabel("Observed |error| / √(2/π)")
    ax.set_title(f"Reliability curve — ECE = {reliability.ece:.4f}")
    ax.grid(alpha=0.3)
    ax.legend()

    # Coverage
    ax = axes[1]
    ax.plot(levels, levels, ":", color="gray", label="ideal")
    ax.plot(levels, coverages, "s-", color="#d62728", markersize=6, linewidth=1.8)
    ax.set_xlabel("Nominal credible level")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"Coverage — NLL = {nll:.3f}")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle(f"Posterior calibration: {result_dir.name}", fontweight="bold")
    out = Path(args.output) if args.output else result_dir / "calibration_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    print(f"ECE = {reliability.ece:.4f}, NLL = {nll:.3f}")
    for L, C in zip(levels, coverages):
        print(f"  nominal {L*100:.0f}% → empirical {C*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
