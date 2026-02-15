#!/usr/bin/env python3
"""
Debug script: run a short 9D DPCC rollout and print obs/action per step.
Use after fixing dt=0.035 to verify policy and env alignment.
  python scripts/dpcc_debug_9d_rollout.py
  python scripts/dpcc_debug_9d_rollout.py --steps 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description="9D DPCC rollout debug: print obs/action per step")
    parser.add_argument("--steps", type=int, default=10, help="Number of env steps")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "third_party"))

    # Load 9D env (same as experiment)
    from enerdynamics.experiments.plugins.environments.d3il_avoiding_9d import D3ILAvoiding9DPlugin
    plugin = D3ILAvoiding9DPlugin()
    env_config = {
        "render": False,
        "horizon": 8,
        "dt": 0.035,
        "obstacle_level": 0,
        "obstacle_radius_by_level": {0: [0.03, 0.025], 1: [0.04, 0.035], 2: [0.06, 0.055]},
        "obstacles": None,
        "robot_radius": 0.01,
    }
    env = plugin.create_env(env_config)
    env._task_env._lazy_init()
    target = np.asarray(env.target, dtype=np.float32).reshape(-1)[:2]

    # Load diffusion + normalizer (same as DPCC plugin)
    import diffuser.utils as dpcc_utils
    dpcc_config_path = project_root / "enerdynamics/solvers/single/dpcc/config/projection_eval.yaml"
    with open(dpcc_config_path, "r", encoding="utf-8") as f:
        dpcc_config = yaml.safe_load(f) or {}
    loadbase, dataset = "logs", "avoiding-d3il-9d"
    loadpath = "diffusion/H8_K20_Dmodels.GaussianDiffusion"
    seed = "5"
    diffusion_experiment = dpcc_utils.load_diffusion(
        loadbase, dataset, loadpath, seed, epoch="best", device=args.device
    )
    diffusion = diffusion_experiment.diffusion
    normalizer = diffusion_experiment.dataset.normalizer

    from enerdynamics.solvers.single.dpcc.patch.avoiding_adapter import AvoidingDPCCAdapter
    indices = {
        "observations": dpcc_config["observation_indices"]["avoiding_9d"],
        "actions": dpcc_config["action_indices"]["avoiding_9d"],
    }
    adapter = AvoidingDPCCAdapter(env, dpcc_config, indices)

    from enerdynamics.solvers.single.dpcc.stepper import DPCCTorchStepper
    stepper = DPCCTorchStepper(diffusion)
    diffusion.p_sample_loop = stepper.p_sample_loop
    from diffuser.sampling.policies import Policy
    policy = Policy(
        model=diffusion,
        normalizer=normalizer,
        preprocess_fns=[],
        test_ret=0.0,
        projector=None,
        trajectory_selection="random",
    )

    obs, action, fixed_z = adapter.reset(seed=None)
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    obs_dim = diffusion.observation_dim  # 11 if trained with target, 9 otherwise
    if obs_dim == 11 and obs.size == 9:
        obs = np.concatenate([np.asarray(target, dtype=np.float32).reshape(-1)[:2], obs], axis=0)
    dist0 = float(np.linalg.norm(obs[2:4] - target)) if obs.size == 11 else float(np.linalg.norm(obs[:2] - target))
    print(f"Target (eval): {target}")
    print(f"Model obs_dim={obs_dim} (11 = goal-conditioned [x_des,y_des,x,y,q1..q7]).")
    # Normalizer ranges (training data stats) for comparison with eval obs/action
    try:
        no = normalizer.normalizers["observations"]
        na = normalizer.normalizers["actions"]
        print(f"Normalizer obs: min={getattr(no, 'mins', None)}, max={getattr(no, 'maxs', None)}")
        print(f"Normalizer act: min={getattr(na, 'mins', None)}, max={getattr(na, 'maxs', None)}")
    except Exception as e:
        print(f"Normalizer info: {e}")
    print(f"Initial obs (full): {obs}")
    print(f"  dist_to_target={dist0:.4f}")
    print(f"Steps: {args.steps} (dt=0.035)")
    print("-" * 60)

    for t in range(args.steps):
        if obs_dim == 11 and obs.size == 9:
            obs = np.concatenate([np.asarray(target, dtype=np.float32).reshape(-1)[:2], obs], axis=0)
        action, samples = policy(conditions={0: obs}, batch_size=4, horizon=8, disable_projection=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)  # unnormalized (physical) action
        if action.size == 7:  # 9D qdot
            limit = float(getattr(env, "control_limit", 1.5))
            action = np.clip(action, -limit, limit)
        obs, success, terminated, info = adapter.step(action, obs, fixed_z)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        dist = float(np.linalg.norm(obs[:2] - target))
        obs_str = np.array2string(obs, precision=4, separator=" ", suppress_small=True)
        act_str = np.array2string(action, precision=4, separator=" ", suppress_small=True)
        print(f"t={t:2d}  obs_9d={obs_str}  action_unnorm_7d={act_str}  dist={dist:.4f}")
        if success or terminated:
            break

    print("-" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
