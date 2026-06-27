#!/usr/bin/env python3
"""Table-3 prior gallery — visualize the robotized bodies each prior produces.

Grid: one row per prior, a few example robotized bodies per row, each a 3D
particle scatter colored by actuator_id (grey = passive). Shows the morphologies
that actually enter co-design, side by side across priors. matplotlib Agg
offscreen (no display / pyvista needed), mirroring render_soft_robot_checkpoints.

Usage:
  python scripts/visualizations/soft_robot/co_design/render_prior_gallery.py \
      --priors pointe:data/asset_banks/pointe shape:data/asset_banks/shape \
               trellis:data/asset_banks/trellis random:data/asset_banks/loco_cpu \
               triposg:data/asset_banks/triposg \
      --n 4 --out results/soft_robot/co_design/figures/table3_prior_gallery.png
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from genedynamics.morphology.asset_bank import load_spec_npz

REPR = {"pointe": "point-cloud", "shape": "implicit-SDF", "trellis": "structured-voxel",
        "random": "procedural", "triposg": "image/mesh"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--priors", nargs="+", required=True, help="name:bank_dir")
    ap.add_argument("--n", type=int, default=4, help="example bodies per prior")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    K_act = 10
    base = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[:K_act]
    cmap = ListedColormap(np.vstack([[0.82, 0.82, 0.82, 1.0], base]))  # 0 -> passive

    rows = [(e.split(":", 1)[0], e.split(":", 1)[1]) for e in args.priors]
    nrow, ncol = len(rows), args.n
    fig = plt.figure(figsize=(2.5 * ncol, 2.7 * nrow))
    for r, (name, bank) in enumerate(rows):
        files = sorted(glob.glob(os.path.join(bank, "robotized", "*.npz")))
        pick = np.linspace(0, len(files) - 1, min(ncol, len(files))).astype(int) if files else []
        for c in range(ncol):
            ax = fig.add_subplot(nrow, ncol, r * ncol + c + 1, projection="3d")
            if c < len(pick):
                spec = load_spec_npz(files[pick[c]])
                pos = np.asarray(spec.particles_x0, np.float32)
                pos = pos - (pos.min(0) + pos.max(0)) / 2          # center
                pos = pos / (np.abs(pos).max() + 1e-6) * 0.6       # fit [-0.75,0.75]
                aid = np.asarray(spec.actuator_id)
                col = np.where(aid < 0, 0, aid + 1)
                # DiffuseBot order: native x, y, z
                ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=col, cmap=cmap,
                           vmin=0, vmax=K_act, s=8, depthshade=True)
            ax.set_xlim3d(-0.75, 0.75); ax.set_ylim3d(-0.75, 0.75); ax.set_zlim3d(-0.75, 0.75)
            ax.set_box_aspect((1, 1, 1)); ax.tick_params(labelsize=5)
        ypos = 1.0 - (r + 0.5) / nrow
        fig.text(0.008, ypos, f"{name}\n({REPR.get(name,'?')})", rotation=90,
                 va="center", ha="center", fontsize=10, weight="bold")
    fig.suptitle("Table 3 — robotized bodies per prior (color = actuator id, grey = passive)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"[gallery] wrote {args.out}  ({nrow} priors x {ncol})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
