#!/usr/bin/env python3
"""
Export 3DGS experiment figures: render comparison, uncertainty heatmap.

Usage:
  python scripts/tasks/3dgs/export_3dgs_figures.py results/3dgs/lego_mbd_iid --output figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def load_trajectory(result_dir: Path) -> Optional[Dict[str, Any]]:
    """Load trajectory data from experiment output."""
    traj_path = result_dir / "level_0" / "seed_0" / "trajectory" / "trajectory.json"
    if traj_path.exists():
        with open(traj_path) as f:
            return json.load(f)
    return None


def render_comparison(
    gt_images: np.ndarray,
    pred_images: np.ndarray,
    output_path: Path,
    n_show: int = 4,
) -> None:
    """Save side-by-side GT vs Pred comparison."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n = min(n_show, gt_images.shape[0], pred_images.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    for i in range(n):
        axes[0, i].imshow(np.clip(gt_images[i], 0, 1))
        axes[0, i].set_title(f"GT {i}")
        axes[0, i].axis("off")
        axes[1, i].imshow(np.clip(pred_images[i], 0, 1))
        axes[1, i].set_title(f"Pred {i}")
        axes[1, i].axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def uncertainty_heatmap(
    samples: List[np.ndarray],
    output_path: Path,
) -> None:
    """Compute per-pixel variance across samples and save heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    arr = np.stack([np.asarray(s) for s in samples], axis=0)
    var = np.var(arr, axis=0)
    if var.ndim == 3:
        var = np.mean(var, axis=-1)
    plt.figure(figsize=(6, 5))
    plt.imshow(var, cmap="hot")
    plt.colorbar(label="Variance")
    plt.title("Posterior variance (uncertainty)")
    plt.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Export 3DGS figures")
    parser.add_argument("result_dir", type=str, help="Path to results")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--n-show", type=int, default=4, help="Number of views to show")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    out_dir = Path(args.output) if args.output else result_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    traj = load_trajectory(result_dir)
    if traj is None:
        print(f"No trajectory found in {result_dir}")
        return 0

    # Try loading images from .npy files first, then from trajectory.json
    renders_dir = result_dir / "level_0" / "seed_0" / "renders"
    gt_npy = renders_dir / "gt_images.npy"
    pred_npy = renders_dir / "pred_images.npy"

    if gt_npy.exists() and pred_npy.exists():
        gt = np.load(gt_npy)
        pred = np.load(pred_npy)
    elif "images" in traj:
        gt = np.asarray(traj.get("images", []))
        pred = np.asarray(traj.get("predicted_images", []))
    else:
        gt = np.array([])
        pred = np.array([])

    if gt.size > 0 and pred.size > 0:
        render_comparison(gt, pred, out_dir / "render_comparison.png", args.n_show)
        print(f"Saved {out_dir / 'render_comparison.png'}")

    samples = traj.get("posterior_samples", [])
    if len(samples) > 1:
        uncertainty_heatmap(samples, out_dir / "uncertainty_heatmap.png")
        print(f"Saved {out_dir / 'uncertainty_heatmap.png'}")

    print(f"Figures exported to {out_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
