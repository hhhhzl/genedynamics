#!/usr/bin/env python3
"""
Render GT/pred/uncertainty panel from saved experiment arrays.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _safe_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.maximum(x, 0.0)
    vmax = float(np.percentile(x, 99.0)) if x.size else 1.0
    vmax = max(vmax, 1e-8)
    return np.clip(x / vmax, 0.0, 1.0)


def _enhance_low_contrast_rgb(img: np.ndarray, min_span: float = 0.15) -> np.ndarray:
    """Stretch contrast for low-dynamic-range images without over-amplifying noise.
    Uses 5th/95th percentiles with a minimum span to avoid color distortion."""
    img = np.asarray(img, dtype=np.float32)
    vmin = float(np.percentile(img, 5.0))
    vmax = float(np.percentile(img, 95.0))
    span = max(vmax - vmin, min_span)
    center = (vmin + vmax) * 0.5
    vmin = center - span * 0.5
    vmax = center + span * 0.5
    return np.clip((img - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True, help="Result directory with npy artifacts")
    parser.add_argument("--output", type=str, default=None, help="Output png path")
    parser.add_argument("--n-show", type=int, default=4, help="How many views to visualize")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    output = Path(args.output) if args.output else (result_dir / "render_comparison_uncertainty.png")

    pred_stack_path = result_dir / "pred_stack.npy"
    mean_pred_path = result_dir / "mean_pred.npy"
    var_pred_path = result_dir / "var_pred.npy"
    gt_path = result_dir / "gt_images.npy"
    if not (pred_stack_path.exists() and mean_pred_path.exists() and var_pred_path.exists() and gt_path.exists()):
        print(f"Missing required npy files in {result_dir}")
        return 1

    pred_stack = np.asarray(np.load(pred_stack_path), dtype=np.float32)
    mean_pred = np.asarray(np.load(mean_pred_path), dtype=np.float32)
    var_pred = np.asarray(np.load(var_pred_path), dtype=np.float32)
    gt_images = np.asarray(np.load(gt_path), dtype=np.float32)

    n_views = int(min(gt_images.shape[0], mean_pred.shape[0]))
    n_show = int(max(1, min(args.n_show, n_views)))

    if var_pred.ndim == 4:
        var_map = np.mean(var_pred, axis=-1)
    else:
        var_map = var_pred
    abs_err = np.mean(np.abs(mean_pred[:n_views] - gt_images[:n_views]), axis=-1)
    has_variance = pred_stack.shape[0] > 1 and np.any(var_map > 1e-10)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable")
        return 1

    fig, axes = plt.subplots(4, n_show, figsize=(4 * n_show, 12))
    if n_show == 1:
        axes = np.asarray(axes).reshape(4, 1)
    for i in range(n_show):
        axes[0, i].imshow(np.clip(gt_images[i], 0.0, 1.0))
        axes[0, i].set_title(f"GT view {i}")
        axes[0, i].axis("off")

        axes[1, i].imshow(_enhance_low_contrast_rgb(mean_pred[i]))
        axes[1, i].set_title(f"Mean pred {i}")
        axes[1, i].axis("off")

        axes[2, i].imshow(_safe_norm(abs_err[i]), cmap="magma")
        axes[2, i].set_title("Abs error")
        axes[2, i].axis("off")

        if has_variance:
            axes[3, i].imshow(_safe_norm(var_map[i]), cmap="hot")
            axes[3, i].set_title("Posterior variance")
        else:
            placeholder = np.full_like(var_map[i], 0.3)
            axes[3, i].imshow(placeholder, cmap="gray", vmin=0, vmax=1)
            axes[3, i].text(0.5, 0.5, "N/A (single sample)", ha="center", va="center", fontsize=10, transform=axes[3, i].transAxes)
            axes[3, i].set_title("Posterior variance")
        axes[3, i].axis("off")

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved {output}")
    print(f"pred_stack shape: {tuple(pred_stack.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
