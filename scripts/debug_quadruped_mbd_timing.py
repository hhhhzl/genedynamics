#!/usr/bin/env python
"""
DEBUG: Detailed time profiling for quadruped MBD planning.

Diagnoses where time is spent: compile vs execute, MJX rollout cost,
GPU utilization, batch effectiveness.

Usage:
  MUJOCO_GL=osmesa MUJOCO_MENAGERIE_PATH=/workspace/genedynamics/third_party/mujoco_menagerie \
  python scripts/debug_quadruped_mbd_timing.py [--config CONFIG] [--quick]

Output: timing breakdown, GPU info, batch stats. Remove this script after debugging.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml


def _gpu_info() -> dict:
    """Collect GPU info if available."""
    out = {}
    try:
        import jax
        devices = jax.devices()
        out["jax_devices"] = [str(d) for d in devices]
        out["jax_default_backend"] = str(jax.default_backend())
        gpu_devices = jax.devices("gpu")
        out["jax_gpu_count"] = len(gpu_devices)
    except Exception as e:
        out["jax_error"] = str(e)
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            out["nvidia_smi"] = r.stdout.strip().split("\n")
        else:
            out["nvidia_smi"] = "n/a"
    except Exception as e:
        out["nvidia_smi_error"] = str(e)
    return out


def main():
    ap = argparse.ArgumentParser(description="Debug quadruped MBD timing")
    ap.add_argument("--config", type=str, default="configs/quadruped/flat/mbd_locomotion_plan_go2.yaml")
    ap.add_argument("--quick", action="store_true", help="Use reduced M_k/Ndiffuse for faster diagnosis")
    args = ap.parse_args()

    # Load config
    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.quick:
        sc = config.get("scheduler_config", {}).get("diffusion_schedulers", [{}])[0]
        sc["M_k"] = 32
        sc["Ndiffuse"] = 8
        config["env_params"]["horizon"] = 48
        config["metrics"] = []
        config["visualizations"] = []
        print("[quick] M_k=32, Ndiffuse=8, horizon=48, no viz")

    # Minimize work: single level, single seed, no viz
    config["obstacle_levels"] = [0]
    config["seeds"] = [0]
    config["visualizations"] = config.get("visualizations", [])
    if "motion_replay" in config.get("visualizations", []):
        config["visualizations"] = [v for v in config["visualizations"] if v != "motion_replay"]
    config["auto_report"] = False

    # Use experiment runner for exact same flow
    from genedynamics.experiments.framework.experiment import ExperimentRunner
    from genedynamics.experiments.framework.config import ExperimentConfig

    allowed = set(ExperimentConfig.__dataclass_fields__.keys())
    cfg = ExperimentConfig(**{k: v for k, v in config.items() if k in allowed})
    runner = ExperimentRunner(cfg)
    try:
        from genedynamics.experiments.runner import register_all_plugins
        register_all_plugins(runner)
    except ImportError:
        pass  # runner may have default plugins

    # GPU info
    gpu_info = _gpu_info()
    print("\n--- GPU / JAX ---")
    for k, v in gpu_info.items():
        print(f"  {k}: {v}")

    # Run 1 (compile + execute)
    print("\n--- Run 1 (compile + execute) ---")
    t0 = time.perf_counter()
    try:
        all1 = runner.run_all()
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    dt1 = time.perf_counter() - t0
    planning1 = all1[0].get("planning_time", 0) if all1 else 0
    print(f"  total_time: {dt1:.3f} s, planning_time (from result): {planning1:.3f} s")

    # Run 2 (execute only)
    config["seeds"] = [1]
    cfg2 = ExperimentConfig(**{k: v for k, v in config.items() if k in allowed})
    runner2 = ExperimentRunner(cfg2)
    try:
        register_all_plugins(runner2)
    except NameError:
        pass
    print("\n--- Run 2 (execute only, no compile) ---")
    t0 = time.perf_counter()
    all2 = runner2.run_all()
    dt2 = time.perf_counter() - t0
    planning2 = all2[0].get("planning_time", 0) if all2 else 0
    print(f"  total_time: {dt2:.3f} s, planning_time (from result): {planning2:.3f} s")

    # Isolated rollout benchmark via minimal plan
    print("\n--- Isolated rollout benchmark ---")
    from genedynamics.core.backends.runtime import RuntimeBackendManager
    from genedynamics.experiments.plugins.methods.mbd import MBDMethodPlugin
    from genedynamics.experiments.common.constraints import create_scheduler_from_config

    RuntimeBackendManager.set_backend(cfg.backend, device=cfg.device)
    env_plugin = runner.registry.get_plugin("environment", cfg.env_name)
    env = env_plugin.create_env(cfg.env_params)
    energy = env_plugin.create_energy()
    scheduler = None
    if cfg.scheduler_config:
        scheduler = create_scheduler_from_config(
            cfg.scheduler_config, cfg.backend,
            method_params=cfg.method_params,
            obstacle_config=cfg.obstacle_config or {},
        )
    ds0 = (cfg.scheduler_config or {}).get("diffusion_schedulers", [{}])[0]
    method_config = {
        **cfg.method_params,
        "horizon": cfg.env_params.get("horizon", 96),
        "dt": cfg.env_params.get("dt", 0.03),
        "Nsample": ds0.get("M_k", 128),
        "Ndiffuse": ds0.get("Ndiffuse", 32),
        "temp_sample": ds0.get("T_k", 0.06),
        "beta0": ds0.get("beta0", 1e-5),
        "betaT": ds0.get("betaT", 1e-2),
        "action_limit": cfg.env_params.get("control_limit", 0.15),
        "scheduler": scheduler,
        "env_plugin": env_plugin,
        "env_name": cfg.env_name,
    }
    planner = MBDMethodPlugin().create_planner(env, energy, method_config)
    be = planner._get_backend_impl()
    start_pos, _ = env.reset(seed=0)
    x0 = np.asarray(start_pos, dtype=np.float32)
    horizon = be.horizon
    act_dim = be.act_dim
    Nsample = be.Nsample

    batch_sizes = [16, 64, 128]
    for bs in batch_sizes:
        dummy = np.random.randn(bs, horizon, act_dim).astype(np.float32) * 0.01
        t0 = time.perf_counter()
        for _ in range(3):
            rews = be._rollout_rewards_batch_fn(x0, dummy)
            import jax
            jax.block_until_ready(rews)
        dt = (time.perf_counter() - t0) / 3
        steps = bs * horizon
        print(f"  batch={bs}: {dt*1000:.0f} ms for {steps} steps -> {dt/steps*1e6:.1f} us/step")

    # Summary
    print("\n--- Summary ---")
    compile_est = max(0, planning1 - planning2) if planning1 and planning2 else 0
    print(f"  Run1 planning_time: {planning1:.2f} s")
    print(f"  Run2 planning_time: {planning2:.2f} s")
    print(f"  Est. compile (Run1-Run2): {compile_est:.2f} s")
    print(f"  Nsample={Nsample}, Ndiffuse={be.Ndiffuse}, horizon={horizon}")
    print(f"  Total MJX steps/plan: ~{(be.Ndiffuse-1)*Nsample*horizon}")
    print("  GPU: run 'nvidia-smi -l 1' in another terminal to observe utilization during Run1/Run2")


if __name__ == "__main__":
    main()
