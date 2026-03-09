#!/usr/bin/env python3
"""
Run full 3DGS experiment: Ours-MBD with multi-seed posterior, metrics, figures.

Achieves expected outcomes:
- PSNR/LPIPS (held-out)
- NLL (corr)
- Posterior samples -> uncertainty heatmap
- Qualitative render comparison

Usage:
  python scripts/3dgs/run_full_experiment.py configs/3dgs/lego_mbd_iid.yaml
  python scripts/3dgs/run_full_experiment.py configs/3dgs/lego_mbd_iid.yaml --n-seeds 8 --output results/3dgs/lego_full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config YAML")
    parser.add_argument("--n-seeds", type=int, default=8, help="Number of seeds for posterior samples")
    parser.add_argument("--output", type=str, default=None, help="Output directory override")
    args = parser.parse_args()

    from genedynamics.experiments.framework import ExperimentConfig
    from genedynamics.experiments.runner import register_all_plugins
    from genedynamics.experiments.framework import ExperimentRunner

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1

    config = ExperimentConfig.from_yaml(config_path)
    if args.output:
        config.output_dir = Path(args.output)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = ExperimentRunner(config)
    register_all_plugins(runner)

    seeds = list(range(args.n_seeds))
    config.obstacle_levels = [0]
    config.seeds = seeds

    print(f"=== 3DGS Full Experiment ===")
    print(f"Config: {config_path}")
    print(f"Seeds: {seeds}")
    print(f"Output: {output_dir}")
    print()

    all_results = []
    for seed in seeds:
        print(f"[seed {seed}] Running...")
        try:
            r = runner.run_single_experiment(0, seed)
            all_results.append(r)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("No results.")
        return 1

    planning_results = [r.get("result", {}) for r in all_results]
    scene_params_list = [pr.get("scene_params") for pr in planning_results if pr.get("scene_params") is not None]
    total_log_probs = [pr.get("total_log_prob") for pr in planning_results if pr.get("total_log_prob") is not None]
    camera_trajectories = [pr.get("camera_trajectory") for pr in planning_results if pr.get("camera_trajectory") is not None]

    env_plugin = runner.registry.get_plugin("environment", config.env_name)
    env = env_plugin.create_env({**config.env_params, "seed": 0})
    obs = env.get_observations()
    gt_images = np.asarray(obs.images, dtype=np.float32)
    camera_poses = np.asarray(obs.camera_poses, dtype=np.float32)
    if camera_poses.ndim == 1:
        camera_poses = camera_poses[None, :]

    method_plugin = runner.registry.get_plugin("method", config.method)
    energy = env_plugin.create_energy(env)
    method_config = {
        **config.method_params,
        "np_random_seed": 0,
        "env_plugin": None,
        "env_name": config.env_name,
        "constraint_pipeline": None,
        "scheduler": None,
    }
    planner = method_plugin.create_planner(env, energy, method_config)
    renderer = (getattr(planner, "config", None) or {}).get("renderer")

    # Prefer GsplatRenderer for eval when CUDA available (real 3DGS PSNR)
    eval_renderer = renderer
    try:
        from genedynamics.solvers.single.mbd3d.implementations.gsplat_renderer import (
            GsplatRenderer,
            GSPLAT_AVAILABLE,
        )
        import torch
        if GSPLAT_AVAILABLE and torch.cuda.is_available():
            h, w = gt_images.shape[1], gt_images.shape[2]
            eval_renderer = GsplatRenderer(
                image_height=h,
                image_width=w,
                image_channels=gt_images.shape[3] if gt_images.ndim > 3 else 3,
            )
            print("Using GsplatRenderer for evaluation (CUDA)")
    except Exception:
        pass

    pred_images_list = []
    for sp in scene_params_list:
        r = eval_renderer if eval_renderer is not None else renderer
        if r is not None:
            try:
                pred = r.render(sp, camera_poses, obs.intrinsics)
                pred_images_list.append(np.asarray(pred, dtype=np.float32))
            except Exception:
                if renderer is not None:
                    pred = renderer.render(sp, camera_poses, obs.intrinsics)
                    pred_images_list.append(np.asarray(pred, dtype=np.float32))
                else:
                    pred_images_list.append(np.zeros_like(gt_images))
        else:
            pred_images_list.append(np.zeros_like(gt_images))

    if not pred_images_list:
        print("No scene_params or renderer.")
        return 1

    pred_stack = np.stack(pred_images_list, axis=0)
    mean_pred = np.mean(pred_stack, axis=0)
    var_pred = np.var(pred_stack, axis=0)
    if var_pred.ndim == 4:
        var_pred = np.mean(var_pred, axis=-1)

    def _psnr(a, b):
        mse = np.mean((np.asarray(a) - np.asarray(b)) ** 2)
        return float(10.0 * np.log10(1.0 / (mse + 1e-10)))

    def _lpips(a, b):
        try:
            import lpips
            loss_fn = lpips.LPIPS(net="alex")
            a_t = np.clip(a, 0, 1).astype(np.float32)
            b_t = np.clip(b, 0, 1).astype(np.float32)
            if a_t.ndim == 3:
                a_t, b_t = a_t[None], b_t[None]
            a_p = (a_t * 2 - 1).transpose(0, 3, 1, 2)
            b_p = (b_t * 2 - 1).transpose(0, 3, 1, 2)
            d = loss_fn(a_p, b_p)
            return float(np.mean(d))
        except ImportError:
            return -1.0

    psnr = _psnr(mean_pred, gt_images)
    lpips_val = _lpips(mean_pred, gt_images)
    nll = -float(np.mean(total_log_probs)) if total_log_probs else None

    metrics = {
        "psnr": psnr,
        "lpips": lpips_val,
        "nll": nll,
        "n_seeds": len(seeds),
        "n_views": gt_images.shape[0],
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics: {json.dumps(metrics, indent=2)}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_show = min(4, gt_images.shape[0])
        fig, axes = plt.subplots(3, n_show, figsize=(4 * n_show, 12))
        if n_show == 1:
            axes = axes[:, None]
        for i in range(n_show):
            axes[0, i].imshow(np.clip(gt_images[i], 0, 1))
            axes[0, i].set_title(f"GT {i}")
            axes[0, i].axis("off")
            axes[1, i].imshow(np.clip(mean_pred[i], 0, 1))
            axes[1, i].set_title(f"Ours-MBD {i}")
            axes[1, i].axis("off")
            axes[2, i].imshow(var_pred[i], cmap="hot")
            axes[2, i].set_title(f"Uncertainty {i}")
            axes[2, i].axis("off")
        plt.suptitle(f"PSNR={psnr:.2f} dB  LPIPS={lpips_val:.4f}" + (f"  NLL={nll:.2f}" if nll else ""))
        plt.tight_layout()
        plt.savefig(output_dir / "render_comparison_uncertainty.png", dpi=150)
        plt.close()
        print(f"Saved {output_dir / 'render_comparison_uncertainty.png'}")

        fig2, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(np.mean(var_pred, axis=0), cmap="hot")
        ax.set_title("Posterior variance (uncertainty heatmap)")
        ax.axis("off")
        plt.colorbar(im, ax=ax, label="Variance")
        plt.tight_layout()
        plt.savefig(output_dir / "uncertainty_heatmap.png", dpi=150)
        plt.close()
        print(f"Saved {output_dir / 'uncertainty_heatmap.png'}")
    except ImportError:
        print("matplotlib not available, skipping figures")

    print(f"\nDone. Output: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
