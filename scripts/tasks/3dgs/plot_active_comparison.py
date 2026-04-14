#!/usr/bin/env python3
"""
Plot exp2B active-selection comparison: test PSNR vs #views for multiple scorers.

Each run directory contains an `active_log.json` produced by
`run_active_selection.py`; we extract (n_views, psnr) trajectories and overlay.

Usage:
  python scripts/tasks/3dgs/plot_active_comparison.py \
      --runs results/3dgs/exp2_active/posterior_variance \
             results/3dgs/exp2_active/random \
             results/3dgs/exp2_active/max_distance \
      --labels Ours Random MaxDist \
      --output results/figures/active_comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


COLOR_MAP = {
    "ours": "#1f77b4",
    "posterior_variance": "#1f77b4",
    "random": "#888888",
    "max_distance": "#d62728",
    "maxdist": "#d62728",
}


def _read_log(run_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = run_dir / "active_log.json"
    with open(path) as f:
        log = json.load(f)
    n_views: List[int] = []
    psnr: List[float] = []
    lpips: List[float] = []
    for r in log["records"]:
        n_views.append(len(r["active_indices"]))
        m = r.get("metrics", {})
        psnr.append(float(m.get("psnr", np.nan)))
        lpips.append(float(m.get("lpips", np.nan)))
    return np.asarray(n_views), np.asarray(psnr), np.asarray(lpips)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    labels = args.labels or [Path(r).name for r in args.runs]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for run, label in zip(args.runs, labels):
        n, psnr, lpips = _read_log(Path(run))
        color = COLOR_MAP.get(label.lower(), None)
        axes[0].plot(n, psnr, "o-", label=label, linewidth=1.8, markersize=6, color=color)
        if np.isfinite(lpips).any():
            axes[1].plot(n, lpips, "s-", label=label, linewidth=1.8, markersize=6, color=color)

    axes[0].set_xlabel("# active views")
    axes[0].set_ylabel("Test PSNR (dB)")
    axes[0].set_title("PSNR vs view budget")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="lower right")

    axes[1].set_xlabel("# active views")
    axes[1].set_ylabel("Test LPIPS")
    axes[1].set_title("LPIPS vs view budget")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.suptitle("Active view selection comparison", fontweight="bold")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
