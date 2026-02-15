#!/usr/bin/env python3
"""
Debug 9D DPCC rollout WITH obstacles: log per-step (x, y, dist_to_target, qdot_norm)
to CSV to analyze overshoot / failure to reach target.
  python scripts/dpcc_debug_9d_with_obstacles.py --out debug_9d_obstacles.csv
  python scripts/dpcc_debug_9d_with_obstacles.py --steps 200 --out debug.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description="9D DPCC rollout with obstacles, log dist/action per step")
    parser.add_argument("--steps", type=int, default=200, help="Max env steps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="", help="CSV path for per-step log (default: print only)")
    parser.add_argument("--level", type=int, default=0, help="Obstacle level")
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "third_party"))

    # Obstacle config (match dpcc_train_eval_9d.yaml)
    obstacle_config = {
        "use_d3il_preset": True,
        "obstacle_radius_by_level": {0: [0.03, 0.025], 1: [0.04, 0.035], 2: [0.06, 0.055]},
        "robot_radius": 0.01,
    }
    from enerdynamics.experiments.plugins.obstacles.d3il_avoiding_fixed import D3ILAvoidingFixedGeneratorPlugin
    start_pos = np.array([0.5, -0.28], dtype=np.float32)
    target_pos = np.array([0.5, 0.35], dtype=np.float32)
    gen = D3ILAvoidingFixedGeneratorPlugin()
    obstacles = gen.generate(args.level, args.seed, start_pos, target_pos, obstacle_config)

    # 9D env with obstacles
    from enerdynamics.experiments.plugins.environments.d3il_avoiding_9d import D3ILAvoiding9DPlugin
    plugin = D3ILAvoiding9DPlugin()
    env_config = {
        "render": False,
        "horizon": 8,
        "dt": 0.035,
        "obstacle_level": args.level,
        "obstacle_radius_by_level": obstacle_config["obstacle_radius_by_level"],
        "obstacles": obstacles,
        "robot_radius": obstacle_config["robot_radius"],
    }
    env = plugin.create_env(env_config)
    env._task_env._lazy_init()
    target = np.asarray(env.target, dtype=np.float32).reshape(-1)[:2]

    # Load diffusion + normalizer
    import diffuser.utils as dpcc_utils
    dpcc_config_path = project_root / "enerdynamics/solvers/single/dpcc/config/projection_eval.yaml"
    with open(dpcc_config_path, "r", encoding="utf-8") as f:
        dpcc_config = yaml.safe_load(f) or {}
    diffusion_experiment = dpcc_utils.load_diffusion(
        "logs", "avoiding-d3il-9d", "diffusion/H8_K20_Dmodels.GaussianDiffusion",
        str(args.seed), epoch="best", device=args.device
    )
    diffusion = diffusion_experiment.diffusion
    normalizer = diffusion_experiment.dataset.normalizer
    obs_dim = diffusion.observation_dim

    from enerdynamics.solvers.single.dpcc.patch.avoiding_adapter import AvoidingDPCCAdapter
    indices = {
        "observations": dpcc_config["observation_indices"]["avoiding_9d"],
        "actions": dpcc_config["action_indices"]["avoiding_9d"],
    }
    adapter = AvoidingDPCCAdapter(env, dpcc_config, indices)

    from enerdynamics.solvers.single.dpcc.stepper import DPCCTorchStepper
    diffusion.p_sample_loop = DPCCTorchStepper(diffusion).p_sample_loop
    from diffuser.sampling.policies import Policy
    policy = Policy(
        model=diffusion, normalizer=normalizer, preprocess_fns=[], test_ret=0.0,
        projector=None, trajectory_selection="random",
    )

    obs, action, fixed_z = adapter.reset(seed=None)
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if obs_dim == 11 and obs.size == 9:
        obs = np.concatenate([np.asarray(target, dtype=np.float32).reshape(-1)[:2], obs], axis=0)
    limit = float(getattr(env, "control_limit", 1.5))

    rows = []
    success = False
    for t in range(args.steps):
        if obs_dim == 11 and obs.size == 9:
            obs = np.concatenate([np.asarray(target, dtype=np.float32).reshape(-1)[:2], obs], axis=0)
        action, _ = policy(conditions={0: obs}, batch_size=4, horizon=8, disable_projection=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size == 7:
            action = np.clip(action, -limit, limit)
        obs, success, terminated, info = adapter.step(action, obs, fixed_z)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        x, y = float(obs[0]), float(obs[1])
        dist = float(np.linalg.norm(np.array([x, y]) - target))
        qdot_norm = float(np.linalg.norm(action))
        rows.append({"t": t, "x": x, "y": y, "dist_to_target": dist, "qdot_norm": qdot_norm})
        if success or terminated:
            break

    if rows:
        min_dist = min(r["dist_to_target"] for r in rows)
        final_dist = rows[-1]["dist_to_target"]
    else:
        min_dist = final_dist = float("nan")

    n_obs = len(getattr(obstacles, "obstacles", [])) or (len(obstacles) if hasattr(obstacles, "__len__") else 0)
    print(f"Obstacles: level={args.level}, n={n_obs}")
    print(f"Target: {target}")
    print(f"Steps: {len(rows)}, success={success}, min_dist={min_dist:.4f}, final_dist={final_dist:.4f}")

    if args.out:
        out_path = Path(args.out)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["t", "x", "y", "dist_to_target", "qdot_norm"])
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows to {out_path}")
    else:
        for r in rows[:5]:
            print(r)
        if len(rows) > 5:
            print(f"... and {len(rows) - 5} more (use --out file.csv to log all)")


if __name__ == "__main__":
    main()
