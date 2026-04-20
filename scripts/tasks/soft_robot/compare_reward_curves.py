"""MRMFMBD main vs CMA-ES vs CEM — reward curves on one figure.

MRMFMBD's x-axis is reverse diffusion step k=K-1..0 (mapped to iteration 0..99).
CMA-ES / CEM's x-axis is generation 0..99. Both have 100 outer iterations,
so a straight overlay is an honest comparison of per-iteration progress.

A secondary sub-plot shows reward vs CUMULATIVE ROLLOUTS to make the compute
asymmetry explicit (MRMFMBD spends 128 rollouts/step at K·M·num_modes, while
cmaes/cem spend popsize·num_modes=32 rollouts/gen).
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS = {
    "MRMFMBD (main, S1+S3)":
        "/workspace/genedynamics/results/co_design/main/crawling_ground/results.json",
    "Diff (4m, env=200, no S3)":
        "/workspace/genedynamics/results/co_design/ablation/no_fidelity_ladder_env200/results.json",
    "Diff (4m, env=100, no S3)":
        "/workspace/genedynamics/results/co_design/ablation/no_fidelity_ladder_env100/results.json",
    "Diff (1m, env=200, no S1+S3)":
        "/workspace/genedynamics/results/co_design/ablation/no_mode_no_fidelity_env200/results.json",
    "CMA-ES  env=100":
        "/workspace/genedynamics/results/co_design/baselines/cmaes_crawling/results.json",
    "CEM     env=100":
        "/workspace/genedynamics/results/co_design/baselines/cem_crawling/results.json",
    "CMA-ES  env=200":
        "/workspace/genedynamics/results/co_design/baselines/cmaes_crawling_env200/results.json",
    "CEM     env=200":
        "/workspace/genedynamics/results/co_design/baselines/cem_crawling_env200/results.json",
}
COLORS = {
    "MRMFMBD (main, S1+S3)":        "#d62728",
    "Diff (4m, env=200, no S3)":    "#e377c2",
    "Diff (4m, env=100, no S3)":    "#ff7f0e",
    "Diff (1m, env=200, no S1+S3)": "#8c564b",
    "CMA-ES  env=100":              "#1f77b4",
    "CEM     env=100":              "#2ca02c",
    "CMA-ES  env=200":              "#6a3d9a",
    "CEM     env=200":              "#17becf",
}


def _extract(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    r = data[0]["result"]
    # MRMFMBD bridge_history[i] = iteration i, mean_env_return
    if r.get("bridge_history"):
        bh = r["bridge_history"]
        iters = np.array([s["k_forward"] for s in bh])
        best   = np.array([s["mean_env_return"] for s in bh])
        mean   = best.copy()   # only mean available
        # Per-iteration rollouts: M_k × num_modes (we saw 32 × 4 = 128 for main).
        # We infer from num_evaluations / len(bh).
        total_rollouts = int(r.get("num_evaluations", 0))
        per_iter = total_rollouts / max(len(bh), 1)
        cum = np.arange(1, len(bh) + 1) * per_iter
    elif r.get("generation_history"):
        gh = r["generation_history"]
        iters = np.array([g["generation"] for g in gh])
        best  = np.array([g["best_return"] for g in gh])
        mean  = np.array([g["mean_return"] for g in gh])
        total_rollouts = int(r.get("num_evaluations", 0))
        per_iter = total_rollouts / max(len(gh), 1)
        cum = np.arange(1, len(gh) + 1) * per_iter
    else:
        iters, best, mean, cum = np.array([]), np.array([]), np.array([]), np.array([])
    # Running-best (monotone) so curves are comparable — baselines report
    # per-gen best that can drop if elitism isn't strict, MRMFMBD reports the
    # noisy per-step mean. Converting to cumulative max gives "best so far".
    running_best = np.maximum.accumulate(best) if best.size else best
    return {
        "iters": iters, "best": best, "running_best": running_best,
        "mean": mean, "cumrollouts": cum,
        "final_return": float(r.get("return_", 0.0)),
        "num_evaluations": int(r.get("num_evaluations", 0)),
        "wall_time": float(r.get("wall_time", 0.0)),
    }


def _replay_disp(path: str) -> float:
    """Re-evaluate best (x, phi) at env=200 / friction=0.45 to get a
    fidelity-invariant displacement number so we can compare across
    different training fidelities on equal footing."""
    import numpy as np
    import json
    import jax.numpy as jnp
    from genedynamics.envs.external.jax_mpm.scene import (
        MPMConfig, build_scene, rollout_return,
    )
    with open(path) as f:
        data = json.load(f)
    r = data[0]["result"]
    x = np.asarray(r["x"], dtype=np.float32)
    phi = np.asarray(r["phi"], dtype=np.float32)
    cfg = MPMConfig(n_grid=64, voxel_dims=(4, 3, 4), scale=50.0,
                    act_strength_base=24.0, shaping_weight=100.0,
                    env_horizon=200)
    scene = build_scene(cfg)
    _, disp, _ = rollout_return(jnp.asarray(x), jnp.asarray(phi),
                                jnp.asarray(0.45, dtype=jnp.float32),
                                scene, cfg, 200)
    return float(disp)


def main():
    extracted = {name: _extract(p) for name, p in RUNS.items()}
    # Add a fidelity-invariant disp (re-roll at env=200, fr=0.45) so env=100 and
    # env=200 trainings can be compared on the same physical quantity.
    for name, p in RUNS.items():
        extracted[name]["disp_at_env200"] = _replay_disp(p)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=140)

    # --- Left: reward vs iteration --------------------------------------------
    ax = axes[0]
    for name, d in extracted.items():
        if d["iters"].size == 0: continue
        c = COLORS[name]
        ax.plot(d["iters"], d["running_best"], "-", lw=2.2, color=c,
                label=f"{name}  (final={d['final_return']:.2f})")
    ax.set_xlabel("outer iteration  (diffusion step / generation)")
    ax.set_ylabel("environment return")
    ax.set_title("Reward vs iteration  (all 3 methods have 100 iters)",
                 fontsize=11, weight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    # --- Right: reward vs cumulative rollouts ---------------------------------
    ax = axes[1]
    for name, d in extracted.items():
        if d["cumrollouts"].size == 0: continue
        c = COLORS[name]
        ax.plot(d["cumrollouts"], d["running_best"], "-", lw=2.2, color=c,
                label=f"{name}  ({d['num_evaluations']:,} rollouts, "
                      f"{d['wall_time']:.0f}s)")
    ax.set_xlabel("cumulative rollouts  (compute budget)")
    ax.set_ylabel("running-best environment return")
    ax.set_title("Reward vs compute  (MRMFMBD = K·M·num_modes; baselines = popsize·num_modes)",
                 fontsize=11, weight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "Co-design reward comparison  —  MRMFMBD  vs  CMA-ES  vs  CEM",
        fontsize=13, weight="bold", y=1.01,
    )
    fig.tight_layout()
    out = "/workspace/genedynamics/results/co_design/reward_comparison.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # Console summary
    print(f"\n{'method':<30} | {'final R':>7} | {'disp@200':>8} | "
          f"{'rollouts':>8} | {'wall':>5}")
    print("-" * 75)
    for name, d in extracted.items():
        print(f"{name:<30} | {d['final_return']:>7.2f} | "
              f"{d['disp_at_env200']:>8.4f} | "
              f"{d['num_evaluations']:>8,} | {d['wall_time']:>5.0f}s")
    print()
    print("NOTE: 'final R' is the training-time return; scale depends on rollout horizon")
    print("      (Σ_t disp_t grows with env_steps), so env=100 vs env=200 R numbers are")
    print("      NOT directly comparable.  'disp@200' is the re-rolled displacement at")
    print("      env=200 / fr=0.45 — fidelity-invariant and fair across methods.")


if __name__ == "__main__":
    main()
