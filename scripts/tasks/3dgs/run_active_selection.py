#!/usr/bin/env python3
"""
Run exp2B active view selection.

Given a scene-reconstruction YAML (Replica / TUM / NeRF Synthetic) and a
choice of scorer (`posterior_variance` | `random` | `max_distance`), this
script:

  1. Builds the candidate pool from the env's full training set.
  2. Seeds the active pool with `--n-init` views.
  3. Runs `--n-rounds` iterations of (score → add → re-run MBD3D).
  4. Writes per-round metrics to `<output>/active_log.json`.

Usage:
  python scripts/tasks/3dgs/run_active_selection.py \
      configs/3dgs/exp2_active/mbd/lego_active.yaml \
      --scorer posterior_variance --n-init 10 --n-rounds 5 --n-select 1 \
      --output results/3dgs/exp2_active/lego_posterior_variance
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Active view selection (exp2B)")
    parser.add_argument("config", type=str, help="Scene reconstruction config YAML")
    parser.add_argument("--scorer", type=str, default="posterior_variance",
                        choices=["posterior_variance", "random", "max_distance"])
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--n-rounds", type=int, default=5)
    parser.add_argument("--n-select", type=int, default=1)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from genedynamics.experiments.framework import ExperimentConfig, ExperimentRunner
    from genedynamics.experiments.runner import register_all_plugins
    from genedynamics.experiments.plugins.methods.mbd3d_active import (
        MBD3DActiveMethodPlugin,
    )

    config = ExperimentConfig.from_yaml(
        Path(args.config) if Path(args.config).is_absolute() else ROOT / args.config
    )
    runner = ExperimentRunner(config)
    register_all_plugins(runner)

    env_plugin = runner.registry.get_plugin("environment", config.env_name)
    env = env_plugin.create_env(config.env_params)
    energy = env_plugin.create_energy(env)

    method_config = {**config.method_params}
    if config.scheduler_config and "scheduler_config" not in method_config:
        method_config["scheduler_config"] = config.scheduler_config

    n_candidates = env.get_observations().num_views
    init_indices = list(range(min(args.n_init, n_candidates)))

    plugin = MBD3DActiveMethodPlugin()
    records, final_scene = plugin.run_active(
        env, energy, method_config,
        scorer_name=args.scorer,
        init_indices=init_indices,
        n_rounds=args.n_rounds,
        n_select_per_round=args.n_select,
        rng_seed=args.seed,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = {
        "config": str(Path(args.config).resolve()),
        "scorer": args.scorer,
        "n_init": args.n_init,
        "n_rounds": args.n_rounds,
        "n_select_per_round": args.n_select,
        "seed": args.seed,
        "n_candidates": int(n_candidates),
        "records": [
            {
                "round": r.round_idx,
                "selected_this_round": list(r.selected_this_round),
                "active_indices": list(r.active_indices),
                "metrics": {k: float(v) for k, v in r.metrics.items()},
            }
            for r in records
        ],
    }
    with open(out_dir / "active_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"Wrote {out_dir / 'active_log.json'}")

    # Short summary
    print("\nRound | n_views | test_psnr")
    for r in records:
        psnr = r.metrics.get("psnr", float("nan"))
        n_active = len(r.active_indices)
        print(f"  {r.round_idx:3d}  | {n_active:6d}  | {psnr:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
