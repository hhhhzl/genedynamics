#!/usr/bin/env python3
"""Render a folder of generated meshes (TripoSG output) as a point-cloud grid.

Surface (quadric) decimation of multi-million-face meshes is too slow, so we
subsample vertices and scatter them — fast and good enough to see the shape the
3D-gen prior produced. No display needed (matplotlib Agg).

Usage:
  python scripts/visualizations/render_meshes.py results/soft_robot/co_design/figures/triposg_meshes.png <mesh_dir> [--points 9000] [--cols 4]
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_png")
    ap.add_argument("mesh_dir")
    ap.add_argument("--points", type=int, default=9000)
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.mesh_dir, "*.glb")))
    n = len(files)
    ncol = args.cols
    nrow = int(np.ceil(n / ncol))
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(3.0 * ncol, 3.0 * nrow))
    for i, f in enumerate(files):
        m = trimesh.load(f, force="mesh")
        v = np.asarray(m.vertices, dtype=np.float32)
        nv = len(v)
        idx = rng.choice(nv, min(args.points, nv), replace=False)
        vs = v[idx]
        vs = vs - vs.mean(0)
        vs = vs / (np.abs(vs).max() + 1e-9)
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        ax.scatter(vs[:, 0], vs[:, 2], vs[:, 1], s=1.4, c=vs[:, 1], cmap="viridis",
                   depthshade=True, edgecolors="none")
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=16, azim=-60)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(f"{os.path.basename(f).replace('.glb','')}\n{nv//1000}k verts", fontsize=8)
        print(f"  {os.path.basename(f)}: {nv} verts, watertight={m.is_watertight}", flush=True)

    fig.suptitle("TripoSG image→3D prior output (vertex point cloud)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)), exist_ok=True)
    fig.savefig(args.out_png, dpi=130)
    print(f"OK: wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
