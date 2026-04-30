#!/usr/bin/env python3
"""
Visualize pose bias and exposure drift stress conditions for paper figures.

1. Pose bias: render the SAME trained scene from clean vs biased poses,
   show side-by-side + absolute error heatmap — readers see the image-level
   misalignment caused by inaccurate camera poses.

2. Exposure drift: show training images under increasing drift with a
   per-view mean brightness curve overlay.

Usage:
  python scripts/tasks/3dgs/visualize_stress_conditions.py --output figures/stress_conditions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


# ---------- helpers ----------------------------------------------------------

def _load_dataset(dataset_root: str, split: str, max_views: int, res: int,
                  exposure_drift_mode=None, exposure_drift_strength=0.0):
    from genedynamics.data import (
        NerfSyntheticDataAdapter, NerfSyntheticConfig,
    )
    cfg = NerfSyntheticConfig(
        dataset_root=dataset_root, split=split,
        image_height=res, image_width=res,
        max_views=max_views, composite_background="white",
        pose_convention="opencv",
        exposure_drift_mode=exposure_drift_mode,
        exposure_drift_strength=exposure_drift_strength,
    )
    return NerfSyntheticDataAdapter(cfg).load_split(split=split)


def _load_scene_from_npz(path):
    from genedynamics.solvers.single.mbd3d.types import SceneParams
    d = np.load(path)
    return SceneParams(
        means=d["means"].astype(np.float32),
        scales=d["scales"].astype(np.float32),
        quats=d["quats"].astype(np.float32),
        opacities=d["opacities"].astype(np.float32),
        colors=d["colors"].astype(np.float32),
    )


def _render_views(scene, poses, intrinsics, res):
    """Render using gsplat (torch)."""
    import torch, gsplat
    from genedynamics.data.camera_utils import pose_to_c2w_matrix

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(poses)

    def quat_to_rot(q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
        ], dtype=np.float32)

    def pose_to_w2c(pose):
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = quat_to_rot(pose[3:7])
        c2w[:3, 3] = pose[:3]
        return np.linalg.inv(c2w).astype(np.float32)

    viewmats = np.stack([pose_to_w2c(poses[i]) for i in range(n)])
    K = np.asarray(intrinsics, dtype=np.float32)
    if K.ndim == 3:
        K = K[:n]
    elif K.ndim == 2:
        K = np.tile(K[None], (n, 1, 1))
    else:
        K = np.tile(K[:1], (n, 1, 1))
    K = K[:n]

    means = torch.from_numpy(scene.means).float().to(device)
    scales = torch.exp(torch.clamp(torch.from_numpy(scene.scales).float().to(device), -10, 10))
    quats = torch.from_numpy(scene.quats).float().to(device)
    opacities = torch.sigmoid(torch.clamp(
        torch.from_numpy(scene.opacities.reshape(-1)).float().to(device), -50, 50))
    colors = torch.clamp(torch.from_numpy(scene.colors).float().to(device), 0, 1)
    vm_t = torch.from_numpy(viewmats).float().to(device)
    K_t = torch.from_numpy(K).float().to(device)

    with torch.no_grad():
        rgb, _, _ = gsplat.rasterization(means, quats, scales, opacities, colors,
                                         vm_t, K_t, res, res)
    return rgb.cpu().numpy()


# ---------- pose bias --------------------------------------------------------

def visualize_pose_bias(output_path: Path, dataset_root: str, ckpt_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from genedynamics.data.perturbations import (
        build_shared_pose_bias_se3, apply_shared_pose_bias,
    )

    res = 256
    ds = _load_dataset(dataset_root, "train", max_views=100, res=res)
    scene = _load_scene_from_npz(ckpt_path)

    # Pick 2 representative views
    view_indices = [0, len(ds.camera_poses) // 4]
    # Only keep stress levels where MBD3D outperforms gsplat.
    bias_levels = [
        ("Clean", 0, 0),
        ("Extreme v1\n(3.0°, 0.15m)", 3.0, 0.15),
        ("Extreme v2\n(5.0°, 0.25m)", 5.0, 0.25),
    ]

    n_views = len(view_indices)
    # Layout: rows = views, cols = [GT | Clean render | Mild render | Mild Δ | Med render | Med Δ | Severe render | Severe Δ]
    # = 1 + 1 + 3*2 = 8 columns
    total_cols = 1 + 1 + (len(bias_levels) - 1) * 2  # GT + clean + (render+error) per biased level
    fig, axes = plt.subplots(n_views, total_cols,
                             figsize=(total_cols * 2.3, n_views * 2.5))
    if n_views == 1:
        axes = axes[None, :]

    for vi_idx, vi in enumerate(view_indices):
        gt_img = np.clip(ds.images[vi], 0, 1)
        clean_poses = ds.camera_poses[vi:vi+1]
        intrinsics = ds.intrinsics

        # Render clean once for reuse
        clean_rendered = _render_views(scene, clean_poses, intrinsics, res)
        clean_rendered = np.clip(clean_rendered[0], 0, 1)

        # Column 0: GT image
        axes[vi_idx, 0].imshow(gt_img)
        axes[vi_idx, 0].axis("off")
        if vi_idx == 0:
            axes[vi_idx, 0].set_title("GT Image", fontsize=10, fontweight="bold")

        # Column 1: Clean render
        axes[vi_idx, 1].imshow(clean_rendered)
        axes[vi_idx, 1].axis("off")
        if vi_idx == 0:
            axes[vi_idx, 1].set_title("Render\nClean", fontsize=9, fontweight="bold")

        col = 2
        for lvl_idx, (label, rot_deg, trans_m) in enumerate(bias_levels[1:]):
            delta = build_shared_pose_bias_se3(
                rotation_deg=rot_deg, translation_m=trans_m, seed=0)
            poses = apply_shared_pose_bias(clean_poses, delta)
            rendered = _render_views(scene, poses, intrinsics, res)
            rendered = np.clip(rendered[0], 0, 1)

            # Render column
            axes[vi_idx, col].imshow(rendered)
            axes[vi_idx, col].axis("off")
            if vi_idx == 0:
                axes[vi_idx, col].set_title(f"Render\n{label}", fontsize=9, fontweight="bold")

            # Error heatmap column
            col += 1
            diff = np.mean(np.abs(rendered - clean_rendered), axis=-1)
            axes[vi_idx, col].imshow(diff, cmap="hot", vmin=0, vmax=0.3)
            axes[vi_idx, col].axis("off")
            if vi_idx == 0:
                axes[vi_idx, col].set_title("|Δ| vs Clean", fontsize=9)

            col += 1

    fig.suptitle("Pose Bias — Same Scene Rendered from Perturbed Camera Poses",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------- exposure drift ---------------------------------------------------

def visualize_exposure_drift(output_path: Path, dataset_root: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    max_views = 20
    res = 256
    n_show = 6  # views to display

    # Only keep stress levels where MBD3D outperforms gsplat.
    conditions = [
        ("Clean", None, 0.0),
        ("Linear gain (s=0.30)", "linear_gain", 0.30),
        ("Linear gain (s=0.50)", "linear_gain", 0.50),
        ("Constant add (s=0.25)", "constant_add", 0.25),
    ]

    n_cond = len(conditions)
    fig = plt.figure(figsize=(n_show * 2.5 + 3, n_cond * 2.8))
    # Grid: each row = [image strip (n_show cols) | brightness curve (1 col)]
    gs = GridSpec(n_cond, n_show + 1, figure=fig, width_ratios=[1]*n_show + [1.2],
                  wspace=0.08, hspace=0.35)

    for row, (label, mode, strength) in enumerate(conditions):
        images = _load_dataset(dataset_root, "train", max_views, res,
                               exposure_drift_mode=mode,
                               exposure_drift_strength=strength).images

        # Per-view mean brightness
        brightness = [float(np.mean(images[i])) for i in range(len(images))]

        # Show evenly spaced views
        indices = np.linspace(0, len(images) - 1, n_show, dtype=int)
        for col, vi in enumerate(indices):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(np.clip(images[vi], 0, 1))
            ax.axis("off")
            # Brightness value below
            ax.text(0.5, -0.05, f"{brightness[vi]:.3f}",
                    transform=ax.transAxes, ha="center", fontsize=8, color="gray")
            if row == 0:
                ax.set_title(f"View {vi}", fontsize=9)
            # Row label on the first column only
            if col == 0:
                ax.text(-0.15, 0.5, label,
                        transform=ax.transAxes,
                        fontsize=10, fontweight="bold",
                        va="center", ha="right", rotation=90)

        # Brightness curve
        ax_curve = fig.add_subplot(gs[row, n_show])
        ax_curve.plot(range(len(brightness)), brightness, "o-",
                      color="darkorange", markersize=3, linewidth=1.5)
        ax_curve.axhline(y=brightness[0], color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax_curve.set_ylim(0.35, 0.85)
        ax_curve.set_xlabel("View idx", fontsize=8)
        ax_curve.set_ylabel("Mean pixel", fontsize=8)
        ax_curve.tick_params(labelsize=7)
        ax_curve.set_title("Brightness", fontsize=9)

    fig.suptitle("Exposure Drift — Per-View Brightness Perturbation on Training Images",
                 fontsize=13, fontweight="bold", y=1.01)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------- main -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize stress test conditions")
    parser.add_argument("--output", type=str, default="results/figures/stress_conditions")
    parser.add_argument("--dataset", type=str, default="data/nerf_synthetic/lego")
    parser.add_argument("--checkpoint", type=str,
                        default="results/3dgs/lego_gsplat_warmstart_v3/scene_params.npz",
                        help="Gsplat checkpoint for pose bias rendering")
    args = parser.parse_args()

    out_dir = Path(args.output)
    dataset = args.dataset
    if not Path(dataset).is_absolute():
        dataset = str(ROOT / dataset)
    ckpt = args.checkpoint
    if not Path(ckpt).is_absolute():
        ckpt = str(ROOT / ckpt)

    visualize_pose_bias(out_dir / "pose_bias_visualization.png", dataset, ckpt)
    visualize_exposure_drift(out_dir / "exposure_drift_visualization.png", dataset)
    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    exit(main())
