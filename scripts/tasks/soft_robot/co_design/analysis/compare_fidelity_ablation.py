"""fidelity-ladder ablation: main (multi-fidelity 30→100→200) vs
fixed-fidelity at env=30 / env=100 / env=200. All 4 methods use mode marginalization
(4-mode marginalization), so it isolates the fidelity ladder.
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
    ("MRMFMBD main (mode marginalization + fidelity ladder ladder 30→100→200)",
     "/workspace/genedynamics/results/soft_robot/main/crawling_ground/results.json",
     "#d62728"),
    ("fixed env=30 (no fidelity ladder)",
     "/workspace/genedynamics/results/soft_robot/ablation/no_fidelity_ladder_env30/results.json",
     "#1f77b4"),
    ("fixed env=100 (no fidelity ladder)",
     "/workspace/genedynamics/results/soft_robot/ablation/no_fidelity_ladder_env100/results.json",
     "#2ca02c"),
    ("fixed env=200 (no fidelity ladder)",
     "/workspace/genedynamics/results/soft_robot/ablation/no_fidelity_ladder_env200/results.json",
     "#ff7f0e"),
]
FRICTIONS = [0.3, 0.4, 0.5, 0.6]


def _load(path):
    with open(path) as f: return json.load(f)[0]["result"]


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


def _corrected_return(r, env_steps, frictions):
    """Reeval at the training fidelity, mode-avg over training frictions."""
    x = np.asarray(r["x"], dtype=np.float32)
    phi = np.asarray(r["phi"], dtype=np.float32)
    cfg = MPMConfig(n_grid=64, voxel_dims=(4, 3, 4), scale=50.0,
                    act_strength_base=24.0, shaping_weight=100.0,
                    env_horizon=env_steps)
    scene = build_scene(cfg)
    rs = []
    for fr in frictions:
        rew, _, _ = rollout_return(jnp.asarray(x), jnp.asarray(phi),
                                   jnp.asarray(fr, dtype=jnp.float32),
                                   scene, cfg, env_steps)
        rs.append(float(rew))
    return float(np.mean(rs))


def _reward_curve(r):
    bh = r.get("bridge_history") or []
    iters = np.array([s["k_forward"] for s in bh])
    vals = np.array([s["mean_env_return"] for s in bh])
    running = np.maximum.accumulate(vals) if vals.size else vals
    total_rollouts = int(r.get("num_evaluations", 0))
    per_iter = total_rollouts / max(len(iters), 1)
    cum = np.arange(1, len(iters) + 1) * per_iter
    return iters, running, cum


# Training fidelity (env steps) per method — used for corrected return reeval.
TRAIN_ENV = {
    "MRMFMBD main (mode marginalization + fidelity ladder ladder 30→100→200)": 200,   # main's eval at fine fid (main uses fine-fidelity validate)
    "fixed env=30 (no fidelity ladder)": 30,
    "fixed env=100 (no fidelity ladder)": 100,
    "fixed env=200 (no fidelity ladder)": 200,
}


def main():
    stats = {}
    for name, path, color in RUNS:
        r = _load(path)
        disps = _cross_mode(r)
        iters, running, cum = _reward_curve(r)
        env_steps = TRAIN_ENV[name]
        corrected_R = _corrected_return(r, env_steps, FRICTIONS)
        # Also compute total physics substeps as a cost metric
        substeps = (r.get("num_evaluations", 0) * env_steps * 16
                    if name != "MRMFMBD main (mode marginalization + fidelity ladder ladder 30→100→200)"
                    else 27_300_000)  # main's known substep count from bridge analysis
        stats[name] = dict(
            color=color, disps=disps, iters=iters, running=running, cum=cum,
            disp_mean=float(disps.mean()), disp_std=float(disps.std()),
            disp_cv=float(disps.std() / disps.mean() * 100),
            disp_min=float(disps.min()), disp_max=float(disps.max()),
            corrected_R=corrected_R,
            raw_R=float(r.get("return_", 0.0)),
            rollouts=int(r.get("num_evaluations", 0)),
            wall=float(r.get("wall_time", 0)),
            substeps=substeps,
        )

    # ---- Reward curve plot ---------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=140)

    ax = axes[0]
    for name, d in stats.items():
        ax.plot(d["iters"], d["running"], "-", lw=2.4, color=d["color"], label=name)
    ax.set_xlabel("outer diffusion step k")
    ax.set_ylabel("environment return (running max)")
    ax.set_title("Reward vs diffusion step  (same K=100 for all)",
                 fontsize=11, weight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.text(0.02, 0.98,
            "note: reward scale depends on env_steps (Σ_t disp_t grows linearly),\n"
            "so envelope heights differ even when gait quality doesn't",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            color="0.35", style="italic")

    ax = axes[1]
    for name, d in stats.items():
        if d["cum"].size == 0: continue
        xs = np.arange(1, len(d["cum"]) + 1) * d["substeps"] / max(len(d["cum"]), 1) / 1e6
        ax.plot(xs, d["running"], "-", lw=2.4, color=d["color"],
                label=f"{name}  ({d['substeps']/1e6:.1f}M substeps, {d['wall']:.0f}s)")
    ax.set_xlabel("cumulative physics substeps  (millions)")
    ax.set_ylabel("running-max environment return")
    ax.set_title("Reward vs compute  (fidelity ladder.s true cost saving)",
                 fontsize=11, weight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "Fidelity ablation — MRMFMBD (fidelity ladder) vs fixed env=30/100/200",
        fontsize=12, weight="bold", y=1.01,
    )
    fig.tight_layout()
    out = "/workspace/genedynamics/results/soft_robot/fidelity_ablation_comparison.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}\n")

    # ---- Console table -------------------------------------------------------
    print("=" * 108)
    print("PER-MODE DISP (evaluated at env=200 rollout, fr ∈ [0.3, 0.4, 0.5, 0.6])")
    print("=" * 108)
    hdr = f"{'method':<42} | " + " | ".join(f"fr={fr}" for fr in FRICTIONS) + " |  mean  |   CV "
    print(hdr); print("-" * len(hdr))
    for name, d in stats.items():
        mark = "★  " if name.startswith("MRMFMBD") else "   "
        print(f"{mark}{name:<39} | " + " | ".join(f"{v:.4f}" for v in d['disps']) +
              f" | {d['disp_mean']:.4f} | {d['disp_cv']:.1f}%")

    print()
    print("=" * 108)
    print("HEADLINE METRICS  (★ = main)")
    print("=" * 108)
    print(f"{'method':<42} | {'mean disp':>9} | {'min disp':>9} | "
          f"{'corr R':>7} | {'rollouts':>8} | {'Msubsteps':>9} | {'wall':>5}")
    print("-" * 108)
    for name, d in stats.items():
        mark = " ★ " if name.startswith("MRMFMBD") else "   "
        print(f"{mark}{name:<39} | "
              f"{d['disp_mean']:>9.4f} | "
              f"{d['disp_min']:>9.4f} | "
              f"{d['corrected_R']:>7.2f} | "
              f"{d['rollouts']:>8,} | "
              f"{d['substeps']/1e6:>9.1f} | "
              f"{d['wall']:>5.0f}s")

    print()
    print("=" * 108)
    print("MAIN vs each fixed-fidelity baseline  (same 12800 rollouts, equal mode marginalization)")
    print("=" * 108)
    main_stats = stats["MRMFMBD main (mode marginalization + fidelity ladder ladder 30→100→200)"]
    for name, d in stats.items():
        if name.startswith("MRMFMBD"): continue
        dmean = (main_stats["disp_mean"] - d["disp_mean"]) / max(d["disp_mean"], 1e-6) * 100
        dmin  = (main_stats["disp_min"]  - d["disp_min"] ) / max(d["disp_min"],  1e-6) * 100
        dsub  = main_stats["substeps"] / max(d["substeps"], 1)
        print(f"  vs {name:<40} : mean disp {dmean:+.0f}%, worst {dmin:+.0f}%, "
              f"cost {dsub:.2f}× theirs")


if __name__ == "__main__":
    main()
