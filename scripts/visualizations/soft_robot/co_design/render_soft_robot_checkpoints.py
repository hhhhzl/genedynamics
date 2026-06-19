#!/usr/bin/env python3
"""G5.1 — Render the co-design evolution grid (DiffuseBot-style, Fig evolution).

Reads a `results_seed_*.json` (its `theta_history`: per-diffusion-step θ in the
optimizer space), re-rolls out a handful of checkpoints on the JAX-MPM scene,
and renders each as a 3D particle point cloud colored by `actuator_id`, in a
horizontal N-column grid showing morphology+controller co-evolving.

No display / PyVista needed: matplotlib Agg offscreen 3D scatter.

Usage:
  python scripts/visualizations/render_soft_robot_checkpoints.py \
      results/soft_robot/co_design/main/crawling_ground/results_seed_0.json \
      results/soft_robot/co_design/figures/evolution.png --voxel-dims 4,3,4 --cols 5 --symmetry z
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import jax.numpy as jnp

from genedynamics.experiments.plugins.task_domains.jax_mpm import JaxMpmTaskDomainProvider
from genedynamics.envs.external.jax_mpm.scene import rollout_with_positions


def _mirror_z(x_half: np.ndarray, vx: int, vy: int, vz_half: int) -> np.ndarray:
    h = x_half.reshape(vx, vy, vz_half)
    return np.concatenate([h, np.flip(h, axis=-1)], axis=-1).reshape(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json", type=str)
    ap.add_argument("out_png", type=str)
    ap.add_argument("--voxel-dims", default="4,3,4")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--symmetry", default="z", choices=["z", "none"])
    ap.add_argument("--friction", type=float, default=0.5)
    ap.add_argument("--num-env-steps", type=int, default=200)
    ap.add_argument("--frame", type=int, default=-1, help="which rollout frame to draw (-1=last)")
    args = ap.parse_args()

    vx, vy, vz = (int(v) for v in args.voxel_dims.split(","))
    x_full_dim = vx * vy * vz

    d = json.load(open(args.results_json))["result"]
    th = [np.asarray(t, dtype=np.float32) for t in d["theta_history"]]
    K = len(th)
    print(f"theta_history: {K} checkpoints, each dim {th[0].shape[0]}")

    # x-block width in optimizer space (half voxels under z-symmetry).
    if args.symmetry == "z":
        x_opt_dim = vx * vy * (vz // 2)
    else:
        x_opt_dim = x_full_dim

    prov = JaxMpmTaskDomainProvider()
    ev = prov.create_evaluator(".", voxel_dims=[vx, vy, vz], n_grid=64,
                               reward_shaping_weight=100.0, act_strength_base=24.0,
                               scale=50.0, task="crawling_ground")
    scene, cfg = ev._scene, ev._mpm_cfg
    aid = np.asarray(scene.actuator_id)  # (N,) -1=passive, 0..K-1 actuators
    K_act = int(cfg.n_actuators)
    fr = jnp.asarray(float(args.friction), jnp.float32)

    # checkpoint indices evenly spaced across the diffusion (last = converged).
    idxs = np.unique(np.linspace(0, K - 1, args.cols).astype(int))

    # color map: passive=light grey, actuators=tab10
    base = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[:K_act]
    cmap = ListedColormap(np.vstack([[0.82, 0.82, 0.82, 1.0], base]))  # index 0 -> passive

    # Pass 1: re-rollout each checkpoint, keep "filled" particles (relative
    # occupancy via mass) at the chosen frame; collect for global bounds.
    panels = []
    for k in idxs:
        theta = th[int(k)]
        x_opt, phi = theta[:x_opt_dim], theta[x_opt_dim:]
        x_full = _mirror_z(x_opt, vx, vy, vz // 2) if args.symmetry == "z" else x_opt
        x_full = np.clip(x_full, 0.2, 1.0).astype(np.float32)
        reward, disp, comx, xhist, massf = rollout_with_positions(
            jnp.asarray(x_full), jnp.asarray(phi), fr, scene, cfg, args.num_env_steps)
        xhist = np.asarray(xhist); massf = np.asarray(massf)
        occ = massf / float(massf.max() + 1e-9)         # per-particle relative occupancy
        keep = occ > 0.45                                # draw the "solid" body only
        panels.append(dict(k=int(k), pos=xhist[args.frame][keep], occ=occ[keep],
                           aid=aid[keep], reward=float(reward), disp=float(disp)))
        print(f"  iter {int(k):3d}: reward={float(reward):+.2f} Δx={float(disp):+.3f} "
              f"filled={int(keep.sum())}/{keep.size}")

    # Global, shared coordinate bounds → forward locomotion is visible across cols.
    allpos = np.concatenate([p["pos"] for p in panels], axis=0)
    lo, hi = allpos.min(0), allpos.max(0)
    pad = 0.04
    xlim = (float(lo[0] - pad), float(hi[0] + pad))
    zlim = (float(lo[2] - pad), float(hi[2] + pad))
    ylim = (float(min(lo[1], 0.0)), float(hi[1] + pad))

    fig = plt.figure(figsize=(4.0 * len(idxs), 4.6))
    for col, p in enumerate(panels):
        ax = fig.add_subplot(1, len(idxs), col + 1, projection="3d")
        pos, occ = p["pos"], p["occ"]
        c = (p["aid"] + 1).clip(0, K_act)               # -1->0 (passive), id->id+1
        s = 6.0 + 34.0 * (occ ** 2)                      # bigger marker for denser voxels
        ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1], c=c, cmap=cmap, vmin=0, vmax=K_act,
                   s=s, depthshade=False, edgecolors="none", alpha=0.95)
        ax.set_xlim(*xlim); ax.set_ylim(*zlim); ax.set_zlim(*ylim)  # shared scale
        ax.set_title(f"iter {p['k']}\nreward={p['reward']:.2f}   Δx={p['disp']:.3f}", fontsize=10)
        ax.set_box_aspect((xlim[1]-xlim[0], zlim[1]-zlim[0], (ylim[1]-ylim[0]) * 1.4))
        ax.view_init(elev=16, azim=-72)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel("x = forward →", fontsize=8, labelpad=-8)

    fig.suptitle(f"Soft-robot co-design evolution — {os.path.basename(args.results_json)}  "
                 f"(particles colored by actuator; shared axes → forward progress)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)), exist_ok=True)
    fig.savefig(args.out_png, dpi=130)
    print(f"OK: wrote {args.out_png}  ({len(idxs)} columns at iters {idxs.tolist()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
