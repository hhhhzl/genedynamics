#!/usr/bin/env python3
"""Render a crawling animation (gif) of a co-designed soft robot.

Re-rolls out a checkpoint's (x, phi) on the JAX-MPM scene, captures particle
positions per env step, and writes a gif of the body locomoting (particles
colored by actuator, fixed world axes so forward progress is visible).

Usage:
  python scripts/visualizations/render_soft_robot_gif.py \
      results/soft_robot/co_design/main/crawling_ground/results_seed_0.json results/soft_robot/co_design/figures/crawl.gif \
      --voxel-dims 4,3,4 --symmetry z --checkpoint -1 --num-env-steps 200 --stride 3 --fps 20
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
import imageio.v2 as imageio
import jax.numpy as jnp

from genedynamics.experiments.plugins.task_domains.jax_mpm import JaxMpmTaskDomainProvider
from genedynamics.envs.external.jax_mpm.scene import rollout_with_positions


def _mirror_z(x_half, vx, vy, vz_half):
    h = x_half.reshape(vx, vy, vz_half)
    return np.concatenate([h, np.flip(h, axis=-1)], axis=-1).reshape(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json")
    ap.add_argument("out_gif")
    ap.add_argument("--voxel-dims", default="4,3,4")
    ap.add_argument("--symmetry", default="z", choices=["z", "none"])
    ap.add_argument("--checkpoint", type=int, default=-1, help="theta_history index (-1=best/final)")
    ap.add_argument("--friction", type=float, default=0.5)
    ap.add_argument("--num-env-steps", type=int, default=200)
    ap.add_argument("--stride", type=int, default=3, help="keep every Nth frame")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    vx, vy, vz = (int(v) for v in args.voxel_dims.split(","))
    x_opt_dim = vx * vy * (vz // 2) if args.symmetry == "z" else vx * vy * vz

    d = json.load(open(args.results_json))["result"]
    th = d["theta_history"]
    theta = np.asarray(th[args.checkpoint], dtype=np.float32)
    x_opt, phi = theta[:x_opt_dim], theta[x_opt_dim:]
    x_full = _mirror_z(x_opt, vx, vy, vz // 2) if args.symmetry == "z" else x_opt
    x_full = np.clip(x_full, 0.2, 1.0).astype(np.float32)

    prov = JaxMpmTaskDomainProvider()
    ev = prov.create_evaluator(".", voxel_dims=[vx, vy, vz], n_grid=64, reward_shaping_weight=100.0,
                               act_strength_base=24.0, scale=50.0, task="crawling_ground")
    scene, cfg = ev._scene, ev._mpm_cfg
    aid = np.asarray(scene.actuator_id)
    K_act = int(cfg.n_actuators)

    reward, disp, comx, xhist, massf = rollout_with_positions(
        jnp.asarray(x_full), jnp.asarray(phi), jnp.asarray(float(args.friction), jnp.float32),
        scene, cfg, args.num_env_steps)
    xhist = np.asarray(xhist); massf = np.asarray(massf)
    occ = massf / float(massf.max() + 1e-9)
    keep = occ > 0.45
    print(f"checkpoint {args.checkpoint}: reward={float(reward):.2f} Δx={float(disp):.3f} "
          f"filled={int(keep.sum())}/{keep.size}, {xhist.shape[0]} frames")

    P = xhist[:, keep, :]                                   # (T, n_keep, 3)
    base = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[:K_act]
    cmap = ListedColormap(np.vstack([[0.82, 0.82, 0.82, 1.0], base]))
    cidx = (aid[keep] + 1).clip(0, K_act)
    s = 8.0 + 32.0 * (occ[keep] ** 2)

    # fixed world bounds across the whole trajectory
    lo, hi = P.reshape(-1, 3).min(0), P.reshape(-1, 3).max(0)
    pad = 0.03
    xlim = (lo[0] - pad, hi[0] + pad); zlim = (lo[2] - pad, hi[2] + pad)
    ylim = (min(lo[1], 0.0), hi[1] + pad)

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    frames = []
    idxs = range(0, P.shape[0], args.stride)
    for t in idxs:
        ax.clear()
        ax.scatter(P[t, :, 0], P[t, :, 2], P[t, :, 1], c=cidx, cmap=cmap, vmin=0, vmax=K_act,
                   s=s, depthshade=False, edgecolors="none", alpha=0.95)
        ax.set_xlim(*xlim); ax.set_ylim(*zlim); ax.set_zlim(*ylim)
        ax.set_box_aspect((xlim[1]-xlim[0], zlim[1]-zlim[0], (ylim[1]-ylim[0]) * 1.6))
        ax.view_init(elev=14, azim=-72)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(f"crawling (iter {args.checkpoint}, reward {float(reward):.1f})  "
                     f"t={t}/{P.shape[0]}", fontsize=10)
        ax.set_xlabel("forward →", fontsize=8, labelpad=-10)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(buf)
    plt.close(fig)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_gif)), exist_ok=True)
    imageio.mimsave(args.out_gif, frames, fps=args.fps, loop=0)
    print(f"OK: wrote {args.out_gif}  ({len(frames)} frames @ {args.fps}fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
