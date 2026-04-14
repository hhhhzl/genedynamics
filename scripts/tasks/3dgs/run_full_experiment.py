#!/usr/bin/env python3
"""
Run MBD3D method experiment (method path only).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _load_scene_from_npz(path: Path):
    from genedynamics.solvers.single.mbd3d.types import SceneParams
    data = np.load(path)
    return SceneParams(
        means=np.asarray(data["means"], dtype=np.float32),
        scales=np.asarray(data["scales"], dtype=np.float32),
        quats=np.asarray(data["quats"], dtype=np.float32),
        opacities=np.asarray(data["opacities"], dtype=np.float32),
        colors=np.asarray(data["colors"], dtype=np.float32),
    )


def _cap_scene_gaussians(scene, max_gaussians: int):
    max_n = int(max_gaussians)
    n = int(np.asarray(scene.means).shape[0])
    if max_n <= 0 or n <= max_n:
        return scene
    # Keep the most opaque Gaussians when warm-start checkpoint is too large for MBD bridge memory.
    opa = np.asarray(scene.opacities, dtype=np.float32).reshape(-1)
    if opa.size != n:
        return scene
    scores = 1.0 / (1.0 + np.exp(-opa))
    keep = np.argsort(-scores)[:max_n]
    keep = np.sort(keep)
    from genedynamics.solvers.single.mbd3d.types import SceneParams

    return SceneParams(
        means=np.asarray(scene.means)[keep],
        scales=np.asarray(scene.scales)[keep],
        quats=np.asarray(scene.quats)[keep],
        opacities=np.asarray(scene.opacities)[keep],
        colors=np.asarray(scene.colors)[keep],
    )


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((np.asarray(a) - np.asarray(b)) ** 2)
    return float(10.0 * np.log10(1.0 / (mse + 1e-10)))


def _lpips(a: np.ndarray, b: np.ndarray, batch_size: int = 16) -> float:
    try:
        import lpips
        import torch

        loss_fn = lpips.LPIPS(net="alex")
        a_t = np.clip(a, 0, 1).astype(np.float32)
        b_t = np.clip(b, 0, 1).astype(np.float32)
        if a_t.ndim == 3:
            a_t, b_t = a_t[None], b_t[None]
        n = a_t.shape[0]
        if torch.cuda.is_available():
            loss_fn = loss_fn.cuda()

        vals = []
        for i in range(0, n, batch_size):
            sl = slice(i, min(i + batch_size, n))
            a_p = (a_t[sl] * 2 - 1).transpose(0, 3, 1, 2)
            b_p = (b_t[sl] * 2 - 1).transpose(0, 3, 1, 2)
            a_pt = torch.from_numpy(np.asarray(a_p)).float()
            b_pt = torch.from_numpy(np.asarray(b_p)).float()
            if torch.cuda.is_available():
                a_pt, b_pt = a_pt.cuda(), b_pt.cuda()
            vals.append(float(loss_fn(a_pt, b_pt).mean().item()))
        return float(np.mean(vals)) if vals else -1.0
    except Exception:
        return -1.0


def _resize_images_to(images: np.ndarray, h: int, w: int) -> np.ndarray:
    arr = np.asarray(images)
    if arr.ndim != 4:
        return arr
    if int(arr.shape[1]) == int(h) and int(arr.shape[2]) == int(w):
        return arr
    try:
        from PIL import Image
    except Exception:
        return arr
    out = []
    for i in range(arr.shape[0]):
        img = np.clip(arr[i], 0.0, 1.0)
        im = Image.fromarray((img * 255).astype(np.uint8))
        rs = np.asarray(im.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
        out.append(rs)
    return np.stack(out, axis=0)


def _save_uncertainty_artifacts(
    output_dir: Path,
    pred_stack: np.ndarray,
    mean_pred: np.ndarray,
    var_pred: np.ndarray,
    gt_images: np.ndarray,
) -> None:
    np.save(output_dir / "pred_stack.npy", np.asarray(pred_stack, dtype=np.float32))
    np.save(output_dir / "mean_pred.npy", np.asarray(mean_pred, dtype=np.float32))
    np.save(output_dir / "var_pred.npy", np.asarray(var_pred, dtype=np.float32))
    np.save(output_dir / "gt_images.npy", np.asarray(gt_images, dtype=np.float32))


def _render_uncertainty_panel(output_dir: Path, n_show: int = 4) -> None:
    panel_script = ROOT / "scripts" / "tasks" / "3dgs" / "render_uncertainty_panel.py"
    if not panel_script.exists():
        return
    try:
        subprocess.run(
            [
                sys.executable,
                str(panel_script),
                "--result-dir",
                str(output_dir),
                "--n-show",
                str(int(n_show)),
            ],
            check=True,
        )
    except Exception:
        # Keep experiment output robust even if figure export fails.
        return


def _configure_jax_memory(
    *,
    preallocate: str,
    allocator: str,
    mem_fraction: float | None,
    cuda_malloc_async: bool,
) -> None:
    prealloc = str(preallocate).lower().strip()
    if prealloc in {"true", "1", "yes", "y"}:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    else:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    alloc = str(allocator).lower().strip()
    if alloc in {"platform", "bfc"}:
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = alloc
    if mem_fraction is not None:
        frac = float(mem_fraction)
        if frac > 0.0:
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(frac)
    if bool(cuda_malloc_async):
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config YAML")
    parser.add_argument("--n-seeds", type=int, default=8, help="Number of posterior seeds")
    parser.add_argument("--output", type=str, default=None, help="Output directory override")
    parser.add_argument("--initial-scene-path", type=str, default=None, help="Optional scene npz for prior-centered initialization")
    parser.add_argument(
        "--max-initial-gaussians",
        type=int,
        default=None,
        help="Optional cap on warm-start Gaussians for VRAM fit; keeps top opacities.",
    )
    parser.add_argument(
        "--jax-preallocate",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Whether JAX preallocates GPU memory. false enables on-demand allocation.",
    )
    parser.add_argument(
        "--jax-allocator",
        type=str,
        default="platform",
        choices=["platform", "bfc"],
        help="JAX allocator backend. platform is more memory-friendly; bfc can be faster.",
    )
    parser.add_argument(
        "--jax-mem-fraction",
        type=float,
        default=None,
        help="Optional JAX memory fraction cap (e.g., 0.5). Ignored when unset.",
    )
    parser.add_argument(
        "--cuda-malloc-async",
        action="store_true",
        help="Enable cuda_malloc_async allocator for improved fragmentation behavior.",
    )
    parser.add_argument("--initialization-mode", type=str, default="prior_center", choices=["prior_center", "direct", "random"])
    parser.add_argument("--init-jitter-scale", type=float, default=1.0)
    parser.add_argument("--best-chain", action="store_true", help="When n_seeds>1, use scene from chain with highest total_log_prob")
    args = parser.parse_args()
    _configure_jax_memory(
        preallocate=args.jax_preallocate,
        allocator=args.jax_allocator,
        mem_fraction=args.jax_mem_fraction,
        cuda_malloc_async=args.cuda_malloc_async,
    )

    from genedynamics.experiments.framework import ExperimentConfig, ExperimentRunner
    from genedynamics.experiments.runner import register_all_plugins
    from genedynamics.data import (
        NerfSyntheticDataAdapter,
        nerf_synthetic_config_from_env_params,
    )

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

    if not isinstance(config.method_params, dict):
        config.method_params = dict(config.method_params or {})
    config.method_params["initialization_mode"] = args.initialization_mode
    config.method_params["init_jitter_scale"] = float(args.init_jitter_scale)

    if args.initial_scene_path:
        ckpt = Path(args.initial_scene_path)
        if not ckpt.is_absolute():
            ckpt = ROOT / ckpt
        if not ckpt.exists():
            print(f"Initial scene not found: {ckpt}")
            return 1
        scene_center = _load_scene_from_npz(ckpt)
        if args.max_initial_gaussians is not None and int(args.max_initial_gaussians) > 0:
            scene_center = _cap_scene_gaussians(scene_center, int(args.max_initial_gaussians))
        config.method_params["initial_scene_center"] = scene_center
        config.method_params["n_gaussians"] = int(scene_center.means.shape[0])

    runner = ExperimentRunner(config)
    register_all_plugins(runner)
    seeds = list(range(int(args.n_seeds)))
    config.obstacle_levels = [0]
    config.seeds = seeds

    all_results = [runner.run_single_experiment(0, seed) for seed in seeds]
    if not all_results:
        print("No results.")
        return 1

    planning_results = [r.get("result", {}) for r in all_results]
    scene_params_list = [pr.get("scene_params") for pr in planning_results if pr.get("scene_params") is not None]
    total_log_probs = [pr.get("total_log_prob") for pr in planning_results if pr.get("total_log_prob") is not None]
    bridge_histories = [pr.get("bridge_history") for pr in planning_results if pr.get("bridge_history") is not None]
    if not scene_params_list:
        print("No scene_params found in results.")
        return 1

    use_best_chain = getattr(args, "best_chain", False) and len(seeds) > 1 and total_log_probs
    if use_best_chain:
        best_idx = int(np.argmax(total_log_probs))
        scene_params_list = [scene_params_list[best_idx]]
        total_log_probs = [total_log_probs[best_idx]]
        bridge_histories = [bridge_histories[best_idx]] if bridge_histories else []

    ep = config.env_params
    train_split = ep.get("split", "train")
    eval_split = ep.get("eval_split", "test")
    eval_resolution = ep.get("eval_resolution") or ep.get("resolution_eval", 512)
    root = Path(ep.get("dataset_root", "")).expanduser()
    if not root.is_absolute():
        root = ROOT / root

    def _make_adapter(split: str) -> "NerfSyntheticDataAdapter":
        cfg = nerf_synthetic_config_from_env_params(
            ep,
            adapter_default_split=split,
            image_height=int(eval_resolution),
            image_width=int(eval_resolution),
        )
        return NerfSyntheticDataAdapter(cfg)

    adapter_eval = _make_adapter(eval_split)
    eval_ds = adapter_eval.load_split(split=eval_split, resolution="eval")
    gt_images = np.asarray(eval_ds.images, dtype=np.float32)
    camera_poses = np.asarray(eval_ds.camera_poses, dtype=np.float32)
    intrinsics = np.asarray(eval_ds.intrinsics, dtype=np.float32)

    adapter_train = _make_adapter(train_split)
    train_ds = adapter_train.load_split(split=train_split, resolution="eval")
    train_gt = np.asarray(train_ds.images, dtype=np.float32)
    train_poses = np.asarray(train_ds.camera_poses, dtype=np.float32)
    train_intrinsics = np.asarray(train_ds.intrinsics, dtype=np.float32)

    method_plugin = runner.registry.get_plugin("method", config.method)
    env_plugin = runner.registry.get_plugin("environment", config.env_name)
    env = env_plugin.create_env({**ep, "seed": 0})
    energy = env_plugin.create_energy(env)
    planner = method_plugin.create_planner(env, energy, {**config.method_params, "np_random_seed": 0})
    renderer = (getattr(planner, "config", None) or {}).get("renderer")
    if renderer is None:
        print("Renderer unavailable from planner.")
        return 1

    render_h = getattr(renderer, "image_height", gt_images.shape[1])
    render_w = getattr(renderer, "image_width", gt_images.shape[2])
    orig_h, orig_w = int(gt_images.shape[1]), int(gt_images.shape[2])
    if (render_h, render_w) != (orig_h, orig_w):
        from genedynamics.data.camera_utils import resize_intrinsics
        def _scale_K(K, oh, ow, nh, nw):
            if K.ndim == 3:
                return np.stack([resize_intrinsics(K[i], oh, ow, nh, nw) for i in range(K.shape[0])], axis=0)
            return resize_intrinsics(K, oh, ow, nh, nw)
        intrinsics = _scale_K(intrinsics, orig_h, orig_w, render_h, render_w)
        if (train_gt.shape[1], train_gt.shape[2]) == (orig_h, orig_w):
            train_intrinsics = _scale_K(train_intrinsics, orig_h, orig_w, render_h, render_w)

    pred_stack = np.stack(
        [np.asarray(renderer.render(sp, camera_poses, intrinsics), dtype=np.float32) for sp in scene_params_list],
        axis=0,
    )
    mean_pred = np.mean(pred_stack, axis=0)
    var_pred = np.var(pred_stack, axis=0)
    if var_pred.ndim == 4:
        var_pred = np.mean(var_pred, axis=-1)
    if gt_images.shape[1:3] != mean_pred.shape[1:3]:
        gt_images = _resize_images_to(gt_images, int(mean_pred.shape[1]), int(mean_pred.shape[2]))

    train_pred_stack = np.stack(
        [np.asarray(renderer.render(sp, train_poses, train_intrinsics), dtype=np.float32) for sp in scene_params_list],
        axis=0,
    )
    train_mean_pred = np.mean(train_pred_stack, axis=0)
    if train_gt.shape[1:3] != train_mean_pred.shape[1:3]:
        train_gt = _resize_images_to(train_gt, int(train_mean_pred.shape[1]), int(train_mean_pred.shape[2]))

    test_psnr = _psnr(mean_pred, gt_images)
    test_mse = float(np.mean((np.asarray(mean_pred) - np.asarray(gt_images)) ** 2))
    train_psnr = _psnr(train_mean_pred, train_gt)
    train_mse = float(np.mean((np.asarray(train_mean_pred) - np.asarray(train_gt)) ** 2))
    train_log_posterior = float(np.mean(total_log_probs)) if total_log_probs else None

    metrics = {
        "test_psnr": test_psnr,
        "test_mse": test_mse,
        "test_lpips": _lpips(mean_pred, gt_images),
        "train_psnr": train_psnr,
        "train_mse": train_mse,
        "train_log_posterior": train_log_posterior,
        "psnr": test_psnr,
        "lpips": _lpips(mean_pred, gt_images),
        "nll": -train_log_posterior if train_log_posterior is not None else None,
        "n_seeds": len(seeds),
        "n_views_test": int(gt_images.shape[0]),
        "n_views_train": int(train_gt.shape[0]),
        "eval_split": eval_split,
        "train_split": train_split,
        "eval_resolution": int(eval_resolution),
        "initialization_mode": args.initialization_mode,
        "best_chain": bool(use_best_chain),
        "perturb_target_split": ep.get("perturb_target_split", ep.get("split", "train")),
        "pose_bias_rotation_deg": float(ep.get("pose_bias_rotation_deg", 0.0)),
        "pose_bias_translation_m": float(ep.get("pose_bias_translation_m", 0.0)),
        "pose_bias_seed": int(ep.get("pose_bias_seed", 0)),
        "exposure_drift_mode": ep.get("exposure_drift_mode"),
        "exposure_drift_strength": float(ep.get("exposure_drift_strength", 0.0)),
    }
    def _json_safe(obj):
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        if isinstance(obj, (float, np.floating)):
            f = float(obj)
            return None if (f != f or np.isinf(f)) else f
        if isinstance(obj, (list, tuple)):
            return [_json_safe(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        return obj

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(_json_safe(metrics), f, indent=2)
    _save_uncertainty_artifacts(output_dir, pred_stack, mean_pred, var_pred, gt_images)
    _render_uncertainty_panel(output_dir, n_show=4)
    if bridge_histories:
        with open(output_dir / "bridge_profile_seed0.json", "w") as f:
            json.dump(bridge_histories[0], f, indent=2)
        summaries = [bh.get("perf_summary") for bh in bridge_histories if isinstance(bh, dict) and bh.get("perf_summary") is not None]
        if summaries:
            agg = {}
            keys = summaries[0].keys()
            for k in keys:
                vals = [float(s[k]) for s in summaries if k in s]
                if vals:
                    agg[k] = float(np.mean(vals))
            with open(output_dir / "bridge_perf_summary.json", "w") as f:
                json.dump(agg, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
