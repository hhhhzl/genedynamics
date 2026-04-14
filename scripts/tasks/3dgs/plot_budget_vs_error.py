#!/usr/bin/env python3
"""
Plot budget-to-error curves (ESS / PSNR vs bridge step count).

Expects each run directory to contain a `bridge_perf_summary.json` or
`bridge_profile_seed0.json` with per-step ESS / log-prob history.

Usage:
  python scripts/tasks/3dgs/plot_budget_vs_error.py \
      --runs results/3dgs/main/clean/lego_mbd_clean \
      --labels "MBD-corr" \
      --output results/figures/budget_vs_error.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _load_bridge_history(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (steps, ess) from the most detailed JSON available."""
    candidates = [
        run_dir / "bridge_profile_seed0.json",
        run_dir / "bridge_perf_summary.json",
    ]
    for p in candidates:
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, dict) and "ess" in data:
            ess = np.asarray(data["ess"], dtype=np.float32)
            steps = np.arange(len(ess))
            return steps, ess
        if isinstance(data, dict) and "per_step" in data:
            per = data["per_step"]
            ess = np.asarray([r.get("ess", np.nan) for r in per], dtype=np.float32)
            return np.arange(len(ess)), ess
    return np.array([]), np.array([])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="Run dirs (one per curve)")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if args.labels is None:
        labels = [Path(r).name for r in args.runs]
    else:
        labels = args.labels

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for run, label in zip(args.runs, labels):
        steps, ess = _load_bridge_history(Path(run))
        if steps.size == 0:
            print(f"[warn] no bridge history for {run}")
            continue
        ax.plot(steps, ess, "-", label=label, linewidth=1.6)

    ax.set_xlabel("Bridge step (budget)")
    ax.set_ylabel("Effective sample size (ESS)")
    ax.set_title("Budget-to-ESS curves")
    ax.grid(alpha=0.3)
    ax.legend()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
