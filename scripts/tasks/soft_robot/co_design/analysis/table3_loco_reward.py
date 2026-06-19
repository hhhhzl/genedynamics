#!/usr/bin/env python3
"""Table-3 locomotion-reward column: how well do a prior's bodies actually move.

For each robotized body in a prior's bank, the MORPHOLOGY is fixed to that body
and only the controller phi is optimized (a short CEM over the JAX-MPM rollout) —
isolating "can the prior's geometry locomote" from controller search. The mean
best-reward over the bank is the prior's downstream locomotion score; paired with
the shape metrics it gives the shape-quality <-> performance correlation that
justifies the (cheap, CPU) shape metrics as a prior-quality proxy.

Usage:
  python scripts/tasks/soft_robot/co_design/analysis/table3_loco_reward.py \
      --bank-dir data/asset_banks/pointe --prior-name pointe \
      --n-bodies 6 --out results/soft_robot/co_design/table3/pointe_loco_reward.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.morphology.asset_bank import load_spec_npz
from genedynamics.envs.external.jax_mpm.scene import (
    MPMConfig, build_scene_from_spec, rollout_return_batch,
)


def _cem_controller(scene, cfg, phi_dim, *, env_steps, n_iter, pop, elite,
                    friction, seed):
    """Short CEM over phi (morphology fixed = full occupancy). Returns best reward."""
    key = jax.random.PRNGKey(seed)
    mean = jnp.zeros((phi_dim,), jnp.float32)
    std = jnp.full((phi_dim,), 0.4, jnp.float32)
    xm = jnp.ones((pop, scene.n_voxels), jnp.float32)          # body at full density
    fr = jnp.full((pop,), float(friction), jnp.float32)
    best = -1e30
    for _ in range(n_iter):
        key, k = jax.random.split(key)
        phis = mean[None, :] + std[None, :] * jax.random.normal(k, (pop, phi_dim))
        rs, _ = rollout_return_batch(xm, phis, fr, scene, cfg, env_steps)
        rs = np.asarray(rs)
        order = np.argsort(-rs)
        elites = np.asarray(phis)[order[:elite]]
        mean = jnp.asarray(elites.mean(0)); std = jnp.asarray(elites.std(0) + 1e-3)
        best = max(best, float(rs.max()))
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank-dir", required=True)
    ap.add_argument("--prior-name", required=True)
    ap.add_argument("--n-bodies", type=int, default=6)
    ap.add_argument("--env-steps", type=int, default=100)
    ap.add_argument("--n-iter", type=int, default=5)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--elite", type=int, default=8)
    ap.add_argument("--friction", type=float, default=1.0)
    ap.add_argument("--n-grid", type=int, default=128, help="MPM background grid (DiffuseBot uses 128)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.bank_dir, "robotized", "*.npz")))
    if not files:
        print(f"[loco] no robotized bodies in {args.bank_dir}", file=sys.stderr)
        return 1
    # deterministic spread over the bank
    idx = np.linspace(0, len(files) - 1, min(args.n_bodies, len(files))).astype(int)
    rewards = []
    for j, i in enumerate(idx):
        spec = load_spec_npz(files[i])
        cfg = MPMConfig(voxel_dims=tuple(int(v) for v in spec.voxel_dims),
                        n_grid=args.n_grid, shaping_weight=100.0, act_strength_base=24.0, scale=50.0)
        scene = build_scene_from_spec(spec, cfg)
        phi_dim = int(cfg.n_actuators * cfg.n_sin_waves + 4 * cfg.n_actuators)
        r = _cem_controller(scene, cfg, phi_dim, env_steps=args.env_steps,
                            n_iter=args.n_iter, pop=args.pop, elite=args.elite,
                            friction=args.friction, seed=args.seed + j)
        rewards.append(r)
        print(f"[loco] {args.prior_name} body {j+1}/{len(idx)} "
              f"({os.path.basename(files[i])}): best_reward={r:.4f}", flush=True)

    rewards = np.asarray(rewards, np.float32)
    out = {"prior": args.prior_name, "n_bodies": len(rewards),
           "mean_reward": float(rewards.mean()), "max_reward": float(rewards.max()),
           "std_reward": float(rewards.std()), "rewards": rewards.tolist()}
    print(f"[loco] {args.prior_name}: mean={out['mean_reward']:.4f} "
          f"max={out['max_reward']:.4f} over {len(rewards)} bodies", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"[loco] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
