#!/usr/bin/env python3
"""
Train 3DGS scene params with gsplat (PyTorch) for 25-35 dB PSNR.

Runs gradient descent on scene params to minimize L2 loss to GT images.
Outputs SceneParams compatible with run_full_experiment evaluation.

Usage:
  python scripts/3dgs/train_gsplat.py configs/3dgs/lego_mbd_iid.yaml --output results/3dgs/lego_gsplat --iters 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load_nerf_synthetic(
    dataset_root: Path,
    split: str = "train",
    max_views: int = 50,
    view_stride: int = 1,
    shuffle_seed: int = 0,
    h: int = 128,
    w: int = 128,
    return_masks: bool = False,
) -> tuple:
    """Load NeRF Synthetic images, poses, intrinsics (no genedynamics deps)."""
    tf_path = dataset_root / f"transforms_{split}.json"
    with open(tf_path) as f:
        meta = json.load(f)
    frames = meta["frames"]
    angle_x = meta["camera_angle_x"]

    stride = max(1, view_stride)
    idx = np.arange(0, len(frames), stride, dtype=np.int32)
    if shuffle_seed is not None:
        rng = np.random.default_rng(int(shuffle_seed))
        rng.shuffle(idx)
    if max_views and idx.size > max_views:
        idx = idx[:max_views]
    idx = np.sort(idx)

    fx = 0.5 * w / np.tan(0.5 * angle_x)
    K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float32)
    opengl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

    def _to_pose(c2w):
        c2w = np.asarray(c2w) @ opengl_to_cv
        t = c2w[:3, 3]
        R = c2w[:3, :3]
        tr = float(np.trace(R))
        if tr > 0:
            s = np.sqrt(tr + 1) * 2
            qw, qx = 0.25 * s, (R[2, 1] - R[1, 2]) / s
            qy, qz = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw, qx = (R[2, 1] - R[1, 2]) / s, 0.25 * s
            qy, qz = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw, qx = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
            qy, qz = 0.25 * s, (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw, qx = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
            qy, qz = (R[1, 2] + R[2, 1]) / s, 0.25 * s
        q = np.array([qw, qx, qy, qz], dtype=np.float32)
        return np.concatenate([t, q / (np.linalg.norm(q) + 1e-8)])

    images, poses, Ks = [], [], []
    masks_list = [] if return_masks else None
    for i in idx:
        fr = frames[int(i)]
        fp = fr.get("file_path", "").replace("\\", "/").lstrip("./")
        path = dataset_root / fp
        for suf in ("", ".png", ".jpg"):
            p = path.with_suffix(suf) if suf else path
            if p.exists():
                path = p
                break
        img = np.asarray(imageio.imread(path), dtype=np.float32) / 255.0
        mask = None
        if img.shape[-1] == 4:
            mask = img[..., 3:4].copy()
            bg = np.ones_like(img[..., :1])
            img = img[..., :3] * img[..., 3:4] + bg * (1 - img[..., 3:4])
        if img.shape[:2] != (h, w):
            from PIL import Image
            img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((w, h))) / 255.0
            if mask is not None:
                m = (mask.squeeze() * 255).astype(np.uint8)
                mask = np.array(Image.fromarray(m).resize((w, h))).astype(np.float32)[..., None] / 255.0
        images.append(img.astype(np.float32))
        if return_masks:
            masks_list.append(mask.astype(np.float32) if mask is not None else np.ones((h, w, 1), dtype=np.float32))
        poses.append(_to_pose(fr["transform_matrix"]))
        Ks.append(K.copy())

    if return_masks and masks_list:
        return np.stack(images), np.stack(poses), np.stack(Ks), np.stack(masks_list)
    return np.stack(images), np.stack(poses), np.stack(Ks)


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
    dataset_root = Path(ep.get("dataset_root", "data/nerf_synthetic/lego"))
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    if not dataset_root.exists():
        dataset_root = ROOT / "data" / "nerf_synthetic" / "lego"
    if not dataset_root.exists():
        dataset_root = ROOT / "data" / "lego"

    max_views = ep.get("max_views", 50)
    res = ep.get("resolution_infer", 128)
    use_mask_loss = args.mask_loss
    out = _load_nerf_synthetic(
        dataset_root,
        split=ep.get("split", "train"),
        max_views=max_views,
        view_stride=ep.get("view_stride", 1),
        shuffle_seed=ep.get("shuffle_seed", 0),
        h=res,
        w=res,
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
                w = masks_t.clamp(0.1, 1.0)
                loss = (diff2 * w).sum() / (w.sum() + 1e-8)
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
                w = masks_t.clamp(0.1, 1.0)
                loss = (diff2 * w).sum() / (w.sum() + 1e-8)
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
    print(f"Final PSNR: {psnr:.2f} dB, {n} Gaussians")

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
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"psnr": psnr, "n_gaussians": int(n), "n_views": n_views}, f, indent=2)
    print(f"Saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
