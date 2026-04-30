"""Post-hoc analysis for non-diffusion baselines (CMAES / CEM).

These methods don't have bridge_history or theta_history — just a final θ* and
per-generation reward history. Produces morphology.png, controller_heatmap.png,
cross_mode_robustness.png/csv, and reward_vs_generation.png.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yaml
import jax.numpy as jnp

from genedynamics.envs.external.jax_mpm.scene import (
    MPMConfig, build_scene, rollout_return, compute_actuation,
)

# Reuse helpers from the diffusion analyzer
from analyze_diffusion_run import (
    VOXEL_DIMS, N_ACT, K_SIN, N_ENV_STEPS, FRICTION_MODES,
    _expand_x, _make_cfg, draw_morphology, draw_controller,
    cross_mode_eval, _phi_to_act_mat,
)


def draw_reward_vs_generation(gen_history: list, out: str, label: str) -> None:
    if not gen_history:
        return
    gens = [g["generation"] for g in gen_history]
    best = [g["best_return"] for g in gen_history]
    mean = [g["mean_return"] for g in gen_history]
    # elite mean is CEM-only; cmaes ignores if missing
    elite = [g.get("elite_mean_return") for g in gen_history]
    has_elite = any(e is not None for e in elite)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=130)
    ax.plot(gens, best, "-", lw=1.8, color="#d62728", label="best return (this gen)")
    ax.plot(gens, mean, "-", lw=1.2, color="#1f77b4", alpha=0.7,
            label="mean return (population)")
    if has_elite:
        ax.plot(gens, [e for e in elite if e is not None],
                "--", lw=1.4, color="#2ca02c", label="elite mean")
    ax.set_xlabel("generation")
    ax.set_ylabel("return (mode-averaged)")
    ax.set_title(f"Reward vs generation  —  {label}", fontsize=11, weight="bold")
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)


def analyze(config_path: str) -> None:
    with open(config_path) as f:
        cfg_data = yaml.safe_load(f) or {}
    out_dir = cfg_data["output_dir"]
    if not os.path.isabs(out_dir):
        out_dir = os.path.join("/workspace/genedynamics", out_dir)
    results_json = os.path.join(out_dir, "results.json")
    if not os.path.exists(results_json):
        print(f"[skip] {config_path}: no results.json")
        return

    label = cfg_data.get("name", os.path.basename(out_dir))
    eval_params = cfg_data.get("evaluator_runtime", {}) or {}
    act_strength = float(eval_params.get("act_strength_base", 24.0))
    backward_pen = float(eval_params.get("backward_penalty_weight", 0.0))

    with open(results_json) as f:
        data = json.load(f)
    r = data[0]["result"]
    x = np.asarray(r["x"], dtype=np.float32)
    phi = np.asarray(r["phi"], dtype=np.float32)
    gen_history = r.get("generation_history", [])

    x_full = _expand_x(x) if x.shape[-1] != 48 else x

    print(f"  [{label}]  return={r.get('return_', 0):.3f}")
    os.makedirs(out_dir, exist_ok=True)

    draw_morphology(x_full, os.path.join(out_dir, "morphology.png"), label)
    draw_controller(phi, os.path.join(out_dir, "controller_heatmap.png"), label)
    cm = cross_mode_eval(x_full, phi, act_strength, backward_pen,
                         os.path.join(out_dir, "cross_mode_robustness.png"),
                         os.path.join(out_dir, "cross_mode_robustness.csv"),
                         label)
    print(f"    cross-mode: mean={cm['mean']:.4f}  CV={cm['cv']:.1f}%  "
          f"min/max={cm['min_over_max']:.2f}")
    if gen_history:
        draw_reward_vs_generation(gen_history,
                                  os.path.join(out_dir, "reward_vs_generation.png"),
                                  label)
        print(f"    reward_vs_generation: {len(gen_history)} generations")
    else:
        print(f"    reward_vs_generation: SKIPPED (no generation_history)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()
    for cfg in args.configs:
        print(f"\n=== {cfg} ===")
        analyze(cfg)


if __name__ == "__main__":
    main()
