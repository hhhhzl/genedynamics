#!/usr/bin/env python3
"""Render robotized soft-body morphologies from one or more asset banks.

Each robotized SoftBodySpec is a particle cloud (particles_x0) colored by
actuator_id. Lays banks out as rows, assets as columns. For the prior-comparison
qualitative grid (Table 3): real 3D-gen prior (TripoSG) vs procedural
(random_shapes).

Usage:
  python scripts/visualizations/render_asset_bank.py results/soft_robot/co_design/figures/asset_banks.png \
      triposg:data/asset_banks/triposg random_shapes:data/asset_banks/loco_cpu --per-row 6
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from genedynamics.morphology.asset_bank import load_spec_npz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_png", type=str)
    ap.add_argument("banks", nargs="+", help="label:bank_dir entries")
    ap.add_argument("--per-row", type=int, default=6)
    args = ap.parse_args()

    rows = []
    for entry in args.banks:
        label, bank_dir = entry.split(":", 1)
        files = sorted(glob.glob(os.path.join(bank_dir, "robotized", "*.npz")))[: args.per_row]
        specs = [load_spec_npz(f) for f in files]
        rows.append((label, specs))
        print(f"{label}: {len(specs)} robotized bodies from {bank_dir}")

    ncol = max(len(s) for _, s in rows)
    nrow = len(rows)
    K_act = 10
    base = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[:K_act]
    cmap = ListedColormap(np.vstack([[0.82, 0.82, 0.82, 1.0], base]))

    fig = plt.figure(figsize=(2.6 * ncol, 2.8 * nrow))
    for r, (label, specs) in enumerate(rows):
        for c in range(ncol):
            ax = fig.add_subplot(nrow, ncol, r * ncol + c + 1, projection="3d")
            if c < len(specs):
                s = specs[c]
                p = np.asarray(s.particles_x0)
                aid = np.asarray(s.actuator_id)
                col = (aid + 1).clip(0, K_act)
                ax.scatter(p[:, 0], p[:, 2], p[:, 1], c=col, cmap=cmap, vmin=0, vmax=K_act,
                           s=7, depthshade=False, edgecolors="none", alpha=0.95)
                ax.set_box_aspect((1, 1, 1))
                ax.view_init(elev=18, azim=-70)
                if c == 0:
                    ax.text2D(-0.15, 0.5, label, transform=ax.transAxes, rotation=90,
                              va="center", ha="center", fontsize=12, fontweight="bold")
                ax.set_title(f"{s.n_particles}p", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            if c >= len(specs):
                ax.set_axis_off()

    fig.suptitle("Robotized morphologies — real 3D-gen prior (TripoSG) vs procedural "
                 "(random_shapes); particles colored by actuator", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)), exist_ok=True)
    fig.savefig(args.out_png, dpi=130)
    print(f"OK: wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
