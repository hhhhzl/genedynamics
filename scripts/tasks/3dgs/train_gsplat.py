#!/usr/bin/env python3
"""
Train 3DGS scene params with gsplat (PyTorch) for 25-35 dB PSNR.

Runs gradient descent on scene params to minimize L2 loss to GT images.
Outputs SceneParams compatible with run_full_experiment evaluation.

Usage:
  python scripts/tasks/3dgs/train_gsplat.py configs/3dgs/lego_mbd_iid.yaml --output results/3dgs/lego_gsplat --iters 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _load_nerf_synthetic_via_adapter(
    ep: dict,
    project_root: Path,
    *,
    split: str,
    h: int,
    w: int,
    max_views: int | None,
    view_stride: int,
    shuffle_seed: int | None,
    return_masks: bool,
) -> tuple:
    """Load NeRF Synthetic via NerfSyntheticDataAdapter (matches MBD stress protocol)."""
    from genedynamics.solvers.single.mbd3d.data import (
        NerfSyntheticDataAdapter,
        nerf_synthetic_config_from_env_params,
    )

    m_ep = dict(ep)
    m_ep["max_views"] = max_views
    m_ep["view_stride"] = view_stride
    m_ep["shuffle_seed"] = shuffle_seed

    dataset_root = Path(m_ep.get("dataset_root", "data/nerf_synthetic/lego"))
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    if not dataset_root.exists():
        fallback = project_root / "data" / "nerf_synthetic" / "lego"
        if fallback.exists():
            dataset_root = fallback
        else:
            alt = project_root / "data" / "lego"
            if alt.exists():
                dataset_root = alt
    m_ep["dataset_root"] = str(dataset_root)

    cfg = nerf_synthetic_config_from_env_params(
        m_ep,
        adapter_default_split=split,
        image_height=h,
        image_width=w,
        return_view_masks=return_masks,
    )
    ds = NerfSyntheticDataAdapter(cfg).load_split(split=split)
    images = ds.images
    poses = ds.camera_poses
    K = ds.intrinsics
    if return_masks and ds.masks is not None:
        return images, poses, K, ds.masks
    if return_masks:
        masks = np.ones((images.shape[0], h, w, 1), dtype=np.float32)
        return images, poses, K, masks
    return images, poses, K


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config YAML")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--iters", type=int, default=8000, help="Training iterations")
    parser.add_argument("--lr", type=float, default=0.0025, help="Learning rate")
    parser.add_argument("--refine-lr", type=float, default=None, help="Lower LR when --from-checkpoint (default 0.0005)")
    parser.add_argument("--n-gaussians", type=int, default=4096, help="Number of Gaussians")
    parser.add_argument("--densify", action="store_true", help="Use gsplat DefaultStrategy")
    parser.add_argument("--max-gaussians", type=int, default=30000, help="Cap Gaussians when densifying")
    parser.add_argument("--mask-loss", action="store_true", help="Weight loss by alpha (foreground focus for 35 dB)")
    parser.add_argument("--from-checkpoint", type=str, default=None, help="Warm start from npz checkpoint")
    args = parser.parse_args()

    import torch
    import gsplat
    import yaml

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ep = cfg.get("env_params", {})
    max_views = ep.get("max_views", 50)
    res = ep.get("resolution_infer", 128)
    use_mask_loss = args.mask_loss
    out = _load_nerf_synthetic_via_adapter(
        ep,
        ROOT,
        split=ep.get("split", "train"),
        h=res,
        w=res,
        max_views=max_views,
        view_stride=int(ep.get("view_stride", 1)),
        shuffle_seed=ep.get("shuffle_seed", 0),
        return_masks=use_mask_loss,
    )
    if use_mask_loss and len(out) == 4:
        images, poses, K, masks = out
    else:
        images, poses, K = out[:3]
        masks = None
    n_views, h, w, c = images.shape

    def quat_to_rot(q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ], dtype=np.float32)

    def pose_to_w2c(pose):
        t, q = pose[:3], pose[3:7]
        R = quat_to_rot(q)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3], c2w[:3, 3] = R, t
        return np.linalg.inv(c2w).astype(np.float32)

    viewmats = np.stack([pose_to_w2c(poses[i]) for i in range(n_views)], axis=0)
    if K.ndim == 2:
        K = np.tile(K[None, :, :], (n_views, 1, 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt = torch.from_numpy(images).float().to(device)
    masks_t = torch.from_numpy(masks).float().to(device) if masks is not None else None

    n = args.n_gaussians
    torch.manual_seed(0)
    use_densify = args.densify
    from_ckpt = args.from_checkpoint
    if from_ckpt:
        ckpt_path = Path(from_ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = ROOT / ckpt_path
        if ckpt_path.exists():
            data = np.load(ckpt_path)
            n = data["means"].shape[0]
            print(f"Loading {n} Gaussians from {ckpt_path}")

    if use_densify:
        if from_ckpt and ckpt_path.exists():
            means = torch.from_numpy(np.asarray(data["means"], dtype=np.float32)).to(device)
            scales = torch.from_numpy(np.asarray(data["scales"], dtype=np.float32)).to(device)
            quats = torch.from_numpy(np.asarray(data["quats"], dtype=np.float32)).to(device)
            opac = np.asarray(data["opacities"]).reshape(-1)
            opacities = torch.from_numpy(opac.astype(np.float32)).to(device)
            colors = torch.from_numpy(np.asarray(data["colors"], dtype=np.float32)).to(device)
        else:
            means = torch.randn(n, 3, device=device) * 0.4
            scales = torch.randn(n, 3, device=device) * 0.1 - 2.0
            quats = torch.randn(n, 4, device=device) * 0.1
            quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)
            opacities = torch.ones(n, device=device) * 0.1
            colors = torch.sigmoid(torch.randn(n, 3, device=device) * 0.5)
        params = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(opacities if opacities.dim() == 1 else opacities.squeeze(-1)),
            "colors": torch.nn.Parameter(colors),
        })
        if not (from_ckpt and ckpt_path.exists()):
            params["quats"].data = params["quats"].data / (params["quats"].data.norm(dim=-1, keepdim=True) + 1e-8)
        lr = args.refine_lr if (args.refine_lr is not None and from_ckpt) else args.lr
        optimizers = {k: torch.optim.Adam([p], lr=lr) for k, p in params.items()}
        strategy = gsplat.DefaultStrategy(
            refine_stop_iter=min(15000, args.iters),
            grow_grad2d=0.0002,
            prune_opa=0.005,
            reset_every=100000,
        )
        strategy.check_sanity(params, optimizers)
        strategy_state = strategy.initialize_state(scene_scale=1.0)
        max_g = args.max_gaussians
        print(f"Training with densification, {n} initial Gaussians, cap={max_g}, {n_views} views, {args.iters} iters")
    else:
        if from_ckpt and ckpt_path.exists():
            means = torch.from_numpy(np.asarray(data["means"], dtype=np.float32)).to(device)
            scales = torch.from_numpy(np.asarray(data["scales"], dtype=np.float32)).to(device)
            quats = torch.from_numpy(np.asarray(data["quats"], dtype=np.float32)).to(device)
            opac = np.asarray(data["opacities"]).reshape(-1, 1)
            opacities = torch.from_numpy(opac.astype(np.float32)).to(device)
            colors = torch.from_numpy(np.asarray(data["colors"], dtype=np.float32)).to(device)
        else:
            means = torch.randn(n, 3, device=device) * 0.4
            scales = torch.randn(n, 3, device=device) * 0.1 - 2.0
            quats = torch.randn(n, 4, device=device) * 0.1
            quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)
            opacities = torch.ones(n, 1, device=device) * 0.1
            colors = torch.sigmoid(torch.randn(n, 3, device=device) * 0.5)
        means.requires_grad_(True)
        scales.requires_grad_(True)
        quats.requires_grad_(True)
        opacities.requires_grad_(True)
        colors.requires_grad_(True)
        lr = args.refine_lr if (args.refine_lr is not None and from_ckpt) else args.lr
        optim = torch.optim.Adam([means, scales, quats, opacities, colors], lr=lr)
        print(f"Training {n} Gaussians, {n_views} views, {args.iters} iters")

    viewmats_t = torch.from_numpy(viewmats).float().to(device)
    Ks_t = torch.from_numpy(K).float().to(device)

    for it in range(args.iters):
        if use_densify:
            for o in optimizers.values():
                o.zero_grad()
            scales_pos = torch.exp(torch.clamp(params["scales"], -10, 10))
            opa = torch.sigmoid(torch.clamp(params["opacities"], -50, 50))
            col = torch.clamp(params["colors"], 0, 1)
            rgb, _, info = gsplat.rasterization(
                params["means"], params["quats"], scales_pos, opa, col,
                viewmats_t, Ks_t, w, h,
                packed=False,
            )
            strategy.step_pre_backward(params, optimizers, strategy_state, it, info)
            diff2 = (rgb - gt) ** 2
            if masks_t is not None:
                weights = masks_t.clamp(0.1, 1.0)
                loss = (diff2 * weights).sum() / (weights.sum() + 1e-8)
            else:
                loss = diff2.mean()
            loss.backward()
            strategy.step_post_backward(params, optimizers, strategy_state, it, info)
            n_curr = params["means"].shape[0]
            if n_curr > max_g:
                with torch.no_grad():
                    opa = torch.sigmoid(params["opacities"].data)
                    _, keep = torch.topk(opa.squeeze(), max_g, largest=True)
                    for k in params:
                        params[k].data = params[k].data[keep]
                    for k in optimizers:
                        optimizers[k] = torch.optim.Adam([params[k]], lr=optimizers[k].param_groups[0]["lr"])
                strategy_state = strategy.initialize_state(scene_scale=1.0)
            for o in optimizers.values():
                o.step()
            means, scales, quats, opacities, colors = params["means"], params["scales"], params["quats"], params["opacities"], params["colors"]
        else:
            optim.zero_grad()
            scales_pos = torch.exp(torch.clamp(scales, -10, 10))
            opa = torch.sigmoid(torch.clamp(opacities, -50, 50)).squeeze(-1)
            col = torch.clamp(colors, 0, 1)
            rgb, _, _ = gsplat.rasterization(
                means, quats, scales_pos, opa, col,
                viewmats_t, Ks_t, w, h,
            )
            diff2 = (rgb - gt) ** 2
            if masks_t is not None:
                weights = masks_t.clamp(0.1, 1.0)
                loss = (diff2 * weights).sum() / (weights.sum() + 1e-8)
            else:
                loss = diff2.mean()
            loss.backward()
            optim.step()

        if (it + 1) % 500 == 0:
            with torch.no_grad():
                mse = ((rgb - gt) ** 2).mean().item()
                psnr = 10.0 * np.log10(1.0 / (mse + 1e-10))
            n_g = params["means"].shape[0] if use_densify else n
            print(f"  iter {it+1}: PSNR={psnr:.2f} dB, N={n_g}", flush=True)

    if use_densify:
        means = params["means"].detach()
        scales = params["scales"].detach()
        quats = params["quats"].detach()
        opacities = params["opacities"].detach().unsqueeze(-1)
        colors = params["colors"].detach()
        n = means.shape[0]
    with torch.no_grad():
        scales_pos = torch.exp(torch.clamp(scales, -10, 10))
        opa = torch.sigmoid(torch.clamp(opacities, -50, 50)).squeeze(-1)
        col = torch.clamp(colors, 0, 1)
        rgb, _, _ = gsplat.rasterization(
            means, quats, scales_pos, opa, col,
            viewmats_t, Ks_t, w, h,
        )
        mse = ((rgb - gt) ** 2).mean().item()
        psnr = 10.0 * np.log10(1.0 / (mse + 1e-10))
    print(f"Final train PSNR: {psnr:.2f} dB, {n} Gaussians")

    # Evaluate on test split (gsplat renderer)
    eval_split = ep.get("eval_split", "test")
    eval_res = ep.get("eval_resolution") or ep.get("resolution_eval", 512)
    test_out = _load_nerf_synthetic_via_adapter(
        ep,
        ROOT,
        split=eval_split,
        h=eval_res,
        w=eval_res,
        max_views=None,
        view_stride=1,
        shuffle_seed=None,
        return_masks=False,
    )
    test_images, test_poses, test_K = test_out[:3]
    test_viewmats = np.stack([pose_to_w2c(test_poses[i]) for i in range(len(test_poses))], axis=0)
    if test_K.ndim == 2:
        test_K = np.tile(test_K[None, :, :], (len(test_poses), 1, 1))
    test_gt = torch.from_numpy(test_images).float().to(device)
    test_viewmats_t = torch.from_numpy(test_viewmats).float().to(device)
    test_Ks_t = torch.from_numpy(test_K).float().to(device)
    n_test = len(test_images)
    batch_size = min(16, n_test)
    with torch.no_grad():
        scales_pos = torch.exp(torch.clamp(scales, -10, 10))
        opa = torch.sigmoid(torch.clamp(opacities, -50, 50)).squeeze(-1)
        col = torch.clamp(colors, 0, 1)
        preds = []
        for i in range(0, n_test, batch_size):
            end = min(i + batch_size, n_test)
            rgb_b, _, _ = gsplat.rasterization(
                means, quats, scales_pos, opa, col,
                test_viewmats_t[i:end], test_Ks_t[i:end], eval_res, eval_res,
            )
            preds.append(rgb_b)
        test_rgb = torch.cat(preds, dim=0)
        test_mse = ((test_rgb - test_gt) ** 2).mean().item()
        test_psnr = 10.0 * np.log10(1.0 / (test_mse + 1e-10))
    print(f"Test PSNR ({eval_split}, {eval_res}x{eval_res}): {test_psnr:.2f} dB")

    train_at_eval_psnr_save = None
    # Sanity: eval on train at eval_res (should be high if rendering is correct)
    if eval_split != "train":
        train_eval_out = _load_nerf_synthetic_via_adapter(
            ep,
            ROOT,
            split="train",
            h=eval_res,
            w=eval_res,
            max_views=max_views,
            view_stride=int(ep.get("view_stride", 1)),
            shuffle_seed=ep.get("shuffle_seed", 0),
            return_masks=False,
        )
        te_imgs, te_poses, te_K = train_eval_out[:3]
        te_viewmats = np.stack([pose_to_w2c(te_poses[i]) for i in range(len(te_poses))], axis=0)
        if te_K.ndim == 2:
            te_K = np.tile(te_K[None, :, :], (len(te_poses), 1, 1))
        te_gt = torch.from_numpy(te_imgs).float().to(device)
        te_vm = torch.from_numpy(te_viewmats).float().to(device)
        te_kk = torch.from_numpy(te_K).float().to(device)
        n_te = len(te_imgs)
        bs = min(16, n_te)
        with torch.no_grad():
            te_preds = []
            for i in range(0, n_te, bs):
                end = min(i + bs, n_te)
                rgb_b, _, _ = gsplat.rasterization(
                    means, quats, scales_pos, opa, col,
                    te_vm[i:end], te_kk[i:end], eval_res, eval_res,
                )
                te_preds.append(rgb_b)
            te_rgb = torch.cat(te_preds, dim=0)
            train_at_eval_psnr = 10.0 * np.log10(1.0 / (((te_rgb - te_gt) ** 2).mean().item() + 1e-10))
        print(f"Train PSNR at {eval_res}x{eval_res} (sanity): {train_at_eval_psnr:.2f} dB")
        train_at_eval_psnr_save = train_at_eval_psnr

    out_dir = Path(args.output or str(ROOT / "results/3dgs/lego_gsplat"))
    out_dir.mkdir(parents=True, exist_ok=True)
    def _to_np(t):
        if torch.is_tensor(t):
            return t.detach().cpu().numpy()
        return np.asarray(t)
    np.savez(
        out_dir / "scene_params.npz",
        means=_to_np(means),
        scales=_to_np(scales),
        quats=_to_np(quats),
        opacities=_to_np(opacities).reshape(-1, 1),
        colors=_to_np(colors),
        camera_poses=poses,
        intrinsics=K,
        gt_images=images,
    )
    metrics_out = {
        "psnr": psnr,
        "train_psnr": psnr,
        "test_psnr": test_psnr,
        "n_gaussians": int(n),
        "n_views": n_views,
        "eval_split": eval_split,
        "eval_resolution": eval_res,
    }
    if train_at_eval_psnr_save is not None:
        metrics_out["train_psnr_at_eval_res"] = train_at_eval_psnr_save
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
