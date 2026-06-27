"""Focused 4-method comparison: MRMFMBD main vs Diff(1m,env=200) vs CMA-ES env=200 vs CEM env=200.

Produces:
  - results/soft_robot/co_design/compare_4_methods.png  — reward curves (left: per-iter; right: per-compute)
  - Console summary table with per-mode disp, mean, CV, worst-case, wall-time
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax.numpy as jnp
from genedynamics.envs.external.jax_mpm.scene import MPMConfig, build_scene, rollout_return


RUNS = [
    ("MRMFMBD main (mode marginalization + fidelity ladder, 4-mode)",
     "/workspace/genedynamics/results/soft_robot/co_design/main/crawling_ground/results.json",
     "#d62728"),
    ("Diff 1-mode, env=200 (no mode marginalization + fidelity ladder)",
     "/workspace/genedynamics/results/soft_robot/co_design/ablation/no_mode_no_fidelity_env200/results.json",
     "#8c564b"),
    ("CMA-ES env=200",
     "/workspace/genedynamics/results/soft_robot/co_design/baselines/cmaes_crawling_env200/results.json",
     "#6a3d9a"),
    ("CEM env=200",
     "/workspace/genedynamics/results/soft_robot/co_design/baselines/cem_crawling_env200/results.json",
     "#17becf"),
]
FRICTIONS = [0.3, 0.4, 0.5, 0.6]


def _load(path):
    with open(path) as f:
        return json.load(f)[0]["result"]


def _cross_mode(r):
    x = np.asarray(r["x"], dtype=np.float32)
    phi = np.asarray(r["phi"], dtype=np.float32)
    cfg = MPMConfig(n_grid=64, voxel_dims=(4, 3, 4), scale=50.0,
                    act_strength_base=24.0, shaping_weight=100.0, env_horizon=200)
    scene = build_scene(cfg)
    disps = []
    for fr in FRICTIONS:
        _, d, _ = rollout_return(jnp.asarray(x), jnp.asarray(phi),
                                 jnp.asarray(fr, dtype=jnp.float32),
                                 scene, cfg, 200)
        disps.append(float(d))
    return np.asarray(disps)


def _reward_curve(r):
    bh = r.get("bridge_history")
    gh = r.get("generation_history")
    if bh:
        iters = np.array([s["k_forward"] for s in bh])
        vals = np.array([s["mean_env_return"] for s in bh])
    elif gh:
        iters = np.array([s["generation"] for s in gh])
        vals = np.array([s["best_return"] for s in gh])
    else:
        iters, vals = np.array([]), np.array([])
    running = np.maximum.accumulate(vals) if vals.size else vals
    total_rollouts = int(r.get("num_evaluations", 0))
    per_iter = total_rollouts / max(len(iters), 1)
    cum = np.arange(1, len(iters) + 1) * per_iter
    return iters, running, cum


def main():
    stats = {}
    for name, path, color in RUNS:
        r = _load(path)
        disps = _cross_mode(r)
        iters, running, cum = _reward_curve(r)
        stats[name] = {
            "color": color,
            "disps": disps,
            "disp_mean": float(disps.mean()),
            "disp_std": float(disps.std()),
            "disp_cv": float(disps.std() / disps.mean() * 100),
            "disp_min": float(disps.min()),
            "disp_max": float(disps.max()),
            "disp_worst_over_best": float(disps.min() / disps.max()),
            "disp_at_045": float(disps[1]) * 0.5 + float(disps[2]) * 0.5,  # avg 0.4/0.5 ≈ 0.45
            "wall_time": float(r.get("wall_time", 0)),
            "rollouts": int(r.get("num_evaluations", 0)),
            "iters": iters,
            "running": running,
            "cum": cum,
        }

    # ---- Plot -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=140)

    ax = axes[0]
    for name, d in stats.items():
        if d["iters"].size == 0: continue
        ax.plot(d["iters"], d["running"], "-", lw=2.4, color=d["color"],
                label=f"{name}")
    ax.set_xlabel("outer iteration  (diffusion step / generation)")
    ax.set_ylabel("environment return (running max)")
    ax.set_title("Reward vs iteration", fontsize=11, weight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    for name, d in stats.items():
        if d["cum"].size == 0: continue
        ax.plot(d["cum"], d["running"], "-", lw=2.4, color=d["color"],
                label=f"{name}  ({d['rollouts']:,} rollouts, {d['wall_time']:.0f}s)")
    ax.set_xlabel("cumulative rollouts  (compute budget)")
    ax.set_ylabel("running-max environment return")
    ax.set_title("Reward vs compute", fontsize=11, weight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "MRMFMBD  vs  Diff (1-mode, env=200)  vs  CMA-ES env=200  vs  CEM env=200",
        fontsize=12, weight="bold", y=1.01,
    )
    fig.tight_layout()
    out = "/workspace/genedynamics/results/soft_robot/co_design/compare_4_methods.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}\n")

    # ---- Console table --------------------------------------------------------
    main_stats = stats["MRMFMBD main (mode marginalization + fidelity ladder, 4-mode)"]

    def _fmt_delta(method_val, main_val, higher_is_better=True):
        """Return a +/- % difference vs main."""
        if main_val == 0: return ""
        delta = (method_val - main_val) / abs(main_val) * 100
        return f"({delta:+.0f}%)"

    print("=" * 96)
    print("PER-MODE DISPLACEMENT (evaluated at env=200 rollout length)")
    print("=" * 96)
    hdr = f"{'method':<36} | " + " | ".join(f"fr={fr}" for fr in FRICTIONS) + " |  mean  |   CV"
    print(hdr); print("-" * len(hdr))
    for name, d in stats.items():
        ds = d["disps"]
        mark = "  ★" if name.startswith("MRMFMBD") else "   "
        row = f"{mark} {name:<33} | " + " | ".join(f"{v:.4f}" for v in ds) + \
              f" | {d['disp_mean']:.4f} | {d['disp_cv']:.1f}%"
        print(row)

    print()
    print("=" * 96)
    print("HEADLINE METRICS  (★ = main, bold = main's advantage)")
    print("=" * 96)
    print(f"{'method':<36} | {'mean disp':>9} | {'min disp':>9} | "
          f"{'CV%':>5} | {'min/max':>7} | {'rollouts':>8} | {'wall':>5}")
    print("-" * 102)
    for name, d in stats.items():
        mark = " ★ " if name.startswith("MRMFMBD") else "   "
        print(f"{mark}{name:<33} | "
              f"{d['disp_mean']:>9.4f} | "
              f"{d['disp_min']:>9.4f} | "
              f"{d['disp_cv']:>5.1f} | "
              f"{d['disp_worst_over_best']:>7.2f} | "
              f"{d['rollouts']:>8,} | "
              f"{d['wall_time']:>5.0f}s")

    print()
    print("=" * 96)
    print("MAIN'S ADVANTAGE vs each baseline (on mean disp across 4 contact modes)")
    print("=" * 96)
    for name, d in stats.items():
        if name.startswith("MRMFMBD"): continue
        delta = (main_stats["disp_mean"] - d["disp_mean"]) / d["disp_mean"] * 100
        min_delta = (main_stats["disp_min"] - d["disp_min"]) / max(d["disp_min"], 1e-6) * 100
        print(f"  vs {name:<36}: "
              f"mean disp  {delta:+.0f}%  ({main_stats['disp_mean']:.4f} vs {d['disp_mean']:.4f}),  "
              f"worst-case {min_delta:+.0f}%  ({main_stats['disp_min']:.4f} vs {d['disp_min']:.4f})")


if __name__ == "__main__":
    main()
