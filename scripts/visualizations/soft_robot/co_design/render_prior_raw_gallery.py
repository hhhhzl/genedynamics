#!/usr/bin/env python3
"""Table-3 prior gallery (RAW native geometry), rendered in DiffuseBot's style.

Matches DiffuseBot's `plot_point_cloud` (point_e/util/plotting.py): a 3D scatter
in native x,y,z order, fixed cubic bounds [-0.75, 0.75], default matplotlib 3D
view, axes box kept. Coloring mirrors DiffuseBot's RGB-per-point look — Point-E
uses its real generated RGB; meshes (Shap-E/TRELLIS/TripoSG, which DiffuseBot did
not have) are surface-sampled and colored by surface normal mapped to RGB so the
colorful point-cloud aesthetic is consistent across priors.

Usage:
  python scripts/visualizations/soft_robot/co_design/render_prior_raw_gallery.py \
      --priors pointe:data/asset_banks/pointe shape:data/asset_banks/shape \
               trellis:data/asset_banks/trellis triposg:data/asset_banks/triposg \
      --n 4 --out results/soft_robot/co_design/figures/table3_prior_raw_gallery.png
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPR = {"pointe": "point-cloud (Point-E)", "shape": "implicit-SDF (Shap-E)",
        "trellis": "structured-voxel (TRELLIS)", "triposg": "image/mesh (TripoSG)",
        "random": "procedural (baseline)"}
FIXED = ((-0.75, -0.75, -0.75), (0.75, 0.75, 0.75))   # DiffuseBot plot_point_cloud default


def _load_points_colors(path, n_sample=4096):
    """Raw asset -> (points (M,3), colors (M,3) in [0,1]), DiffuseBot RGB-style."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        d = np.load(path, allow_pickle=True)
        if "points" in d.files:
            pts = np.asarray(d["points"], np.float32)
            rgb = np.asarray(d["rgb"], np.float32) if "rgb" in d.files else None
            if rgb is None or rgb.max() <= 1e-6:
                rgb = _height_rgb(pts)
            return pts, np.clip(rgb, 0, 1)
        if "centers" in d.files:                        # SplatFlow 3DGS: opacity-filter
            pts = np.asarray(d["centers"], np.float32)
            op = np.asarray(d["opacities"], np.float32).ravel() if "opacities" in d.files else None
            if op is not None:
                keep = op > 0.5
                pts = pts[keep] if keep.sum() > 32 else pts[op > np.quantile(op, 0.97)]
            if len(pts) > 8:
                m = np.linalg.norm(pts - np.median(pts, 0), axis=1)
                pts = pts[m < np.quantile(m, 0.92)]
            return pts, _height_rgb(pts)
        if "voxel_id" in d.files:                       # robotized fallback
            from genedynamics.morphology.asset_bank import load_spec_npz
            pts = np.asarray(load_spec_npz(path).particles_x0, np.float32)
            return pts, _height_rgb(pts)
    import trimesh
    m = trimesh.load(path, force="mesh")
    pts, fid = trimesh.sample.sample_surface(m, n_sample)
    pts = np.asarray(pts, np.float32)
    nrm = np.asarray(m.face_normals[fid], np.float32)   # normal -> RGB (colorful, principled)
    return pts, np.clip((nrm + 1.0) * 0.5, 0, 1)


def _height_rgb(pts):
    h = pts[:, 1]
    h = (h - h.min()) / (np.ptp(h) + 1e-6)
    return plt.get_cmap("viridis")(h)[:, :3]


def _fit(pts, half=0.6):
    """Center + scale so the body fits inside the fixed [-0.75,0.75] cube."""
    pts = pts - (pts.min(0) + pts.max(0)) / 2
    pts = pts / (np.abs(pts).max() + 1e-6) * half
    return pts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--priors", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rows = [(e.split(":", 1)[0], e.split(":", 1)[1]) for e in args.priors]
    nrow, ncol = len(rows), args.n
    fig = plt.figure(figsize=(3.0 * ncol, 3.1 * nrow))
    for r, (name, bank) in enumerate(rows):
        files = sorted(f for ext in ("raw/*.npz", "raw/*.obj", "raw/*.glb",
                                     "meshes/*.obj", "meshes/*.glb", "meshes/*.ply",
                                     "robotized/*.npz")
                       for f in glob.glob(os.path.join(bank, ext)))
        raw = [f for f in files if "/raw/" in f or "/meshes/" in f] or files
        pick = np.linspace(0, len(raw) - 1, min(ncol, len(raw))).astype(int) if raw else []
        for c in range(ncol):
            ax = fig.add_subplot(nrow, ncol, r * ncol + c + 1, projection="3d")
            if c < len(pick):
                try:
                    pts, col = _load_points_colors(raw[pick[c]])
                    pts = _fit(pts)
                    # DiffuseBot order: native x, y, z
                    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=col, s=3, depthshade=True)
                except Exception:
                    ax.text2D(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center")
            ax.set_xlim3d(FIXED[0][0], FIXED[1][0])      # fixed cubic bounds (DiffuseBot)
            ax.set_ylim3d(FIXED[0][1], FIXED[1][1])
            ax.set_zlim3d(FIXED[0][2], FIXED[1][2])
            ax.set_box_aspect((1, 1, 1))
            ax.tick_params(labelsize=5)                  # keep the axes box (DiffuseBot style)
        ypos = 1.0 - (r + 0.5) / nrow
        fig.text(0.008, ypos, REPR.get(name, name), rotation=90, va="center",
                 ha="center", fontsize=11, weight="bold")
    fig.suptitle("Table 3 — native generated geometry per prior (DiffuseBot plot_point_cloud style)",
                 fontsize=13)
    fig.tight_layout(rect=[0.025, 0, 1, 0.97])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"[raw-gallery] wrote {args.out}  ({nrow} priors x {ncol})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
