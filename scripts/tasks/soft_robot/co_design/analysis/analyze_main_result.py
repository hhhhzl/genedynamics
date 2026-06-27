"""Post-hoc analysis of the main co-design result. Generates four figures
plus a cross-mode robustness table — all from the saved best (x*, φ*) and
bridge_history, no retraining required.

Outputs written next to the main results JSON:
  1. morphology.png            — 3D voxel render of the optimized body
  2. controller_heatmap.png    — actuator × time activation heatmap + envelope
  3. cross_mode_robustness.{png,csv} — best θ evaluated at each friction mode
  4. fidelity_budget.png       — FLOPs breakdown per fidelity level
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import jax.numpy as jnp
from genedynamics.envs.external.jax_mpm.scene import (
    MPMConfig, build_scene, rollout_return, compute_actuation,
)


RESULTS_DIR = "/workspace/genedynamics/results/soft_robot/co_design/main/crawling_ground"
VOXEL_DIMS = (4, 3, 4)
N_ACT = 10
K_SIN = 4
N_ENV_STEPS = 200
FRICTION_MODES = [0.3, 0.4, 0.5, 0.6]


def _load_result():
    with open(os.path.join(RESULTS_DIR, "results.json")) as f:
        data = json.load(f)
    r = data[0]["result"]
    return (
        np.asarray(r["x"], dtype=np.float32),
        np.asarray(r["phi"], dtype=np.float32),
        r["bridge_history"],
        float(r["return_"]),
        float(r["wall_time"]),
    )


# ---------------------------------------------------------------------------
# 1. Morphology 3D voxel render
# ---------------------------------------------------------------------------
def draw_morphology(x_full: np.ndarray, out: str) -> None:
    vx, vy, vz = VOXEL_DIMS
    grid = x_full.reshape(vx, vy, vz)

    fig = plt.figure(figsize=(12, 5), dpi=140)
    cmap = plt.get_cmap("viridis")

    def cube_verts(cx, cy, cz, s=0.92):
        h = s / 2
        d = [-h, h]
        return np.array([[cx + dx, cy + dy, cz + dz]
                         for dx in d for dy in d for dz in d])
    faces_idx = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
                 [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]

    def _render(ax, grid_arr, title):
        gap = 0.08
        for i in range(vx):
            for j in range(vy):
                for k in range(vz):
                    cx = i * (1 + gap); cy = j * (1 + gap); cz = k * (1 + gap)
                    v = cube_verts(cx, cy, cz)
                    occ = grid_arr[i, j, k]
                    color = cmap(occ)
                    alpha = 0.25 + 0.70 * (occ - 0.2) / 0.8
                    faces = [[v[p] for p in face] for face in faces_idx]
                    poly = Poly3DCollection(
                        faces, facecolors=color, edgecolors="0.3",
                        linewidths=0.4, alpha=float(np.clip(alpha, 0.1, 0.95)),
                    )
                    ax.add_collection3d(poly)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel("x (forward)", labelpad=-5)
        ax.set_ylabel("y (up)", labelpad=-5)
        ax.set_zlabel("z", labelpad=-5)
        ax.view_init(elev=22, azim=-60)
        ax.set_title(title, fontsize=11, weight="bold")

    ax1 = fig.add_subplot(121, projection="3d")
    prior_grid = np.full(grid.shape, 0.6, dtype=np.float32)
    _render(ax1, prior_grid, "prior (uniform 0.6)")

    ax2 = fig.add_subplot(122, projection="3d")
    _render(ax2, grid, "optimized  x*")

    # shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.2, 1.0))
    cax = fig.add_axes([0.93, 0.25, 0.015, 0.5])
    cb = plt.colorbar(sm, cax=cax)
    cb.set_label("voxel occupancy")

    stats = (f"occupancy  min={grid.min():.2f}  mean={grid.mean():.2f}  max={grid.max():.2f}\n"
             f"voxels at min bound (0.2):  {int(np.isclose(grid, 0.2, atol=0.02).sum())}/{grid.size}    "
             f"voxels at max bound (1.0):  {int(np.isclose(grid, 1.0, atol=0.02).sum())}/{grid.size}")
    fig.text(0.5, 0.02, stats, ha="center", fontsize=9, color="0.25", family="monospace")

    fig.suptitle(f"Morphology  x  —  {vx}×{vy}×{vz} voxel grid  (z-symmetric)",
                 fontsize=13, weight="bold")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# 2. Controller activation heatmap
# ---------------------------------------------------------------------------
def draw_controller(phi: np.ndarray, out: str) -> None:
    # Compute activation without velocity feedback (v_com_x=0 — isolates the
    # periodic structure from the closed-loop dynamics). We use the same
    # compute_actuation function to stay faithful to what the controller
    # really does at v=0.
    cfg = MPMConfig(n_grid=64, voxel_dims=VOXEL_DIMS, env_horizon=N_ENV_STEPS)
    act_mat = np.zeros((N_ACT, N_ENV_STEPS), dtype=np.float32)
    for t in range(N_ENV_STEPS):
        a = compute_actuation(
            jnp.asarray(phi, dtype=jnp.float32),
            jnp.asarray(t, dtype=jnp.int32),
            cfg,
            v_com_x=jnp.asarray(0.0, dtype=jnp.float32),
        )
        act_mat[:, t] = np.asarray(a)

    fig = plt.figure(figsize=(11, 6), dpi=140)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.0, 1.0], hspace=0.55)

    # --- top: heatmap ---
    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(act_mat, aspect="auto", cmap="RdBu_r",
                    vmin=-1, vmax=1, origin="lower",
                    extent=[0, N_ENV_STEPS * 8e-3, -0.5, N_ACT - 0.5])
    ax1.set_yticks(range(N_ACT))
    ax1.set_yticklabels([f"{i}" for i in range(N_ACT)])
    ax1.set_ylabel("actuator index\n(0=back · 9=front)")
    ax1.set_xlabel("time (s)")
    ax1.set_title(r"Actuator output  $a_i(t)$  (at $v^{com}_x = 0$)",
                  fontsize=11, weight="bold", loc="left")
    cb = plt.colorbar(im, ax=ax1, fraction=0.035, pad=0.01)
    cb.set_label(r"$a_i(t)$", fontsize=9)

    # --- middle: per-actuator envelope ---
    ax2 = fig.add_subplot(gs[1])
    # phi layout: W(40) | b(10) | g(10) | a_env(10) | c_env(10)
    a_env = phi[60:70]
    c_env = phi[70:80]
    t_frac = np.arange(N_ENV_STEPS) / N_ENV_STEPS
    envs = 1.0 / (1.0 + np.exp(-(a_env[None, :] + c_env[None, :] * t_frac[:, None])))
    act_cmap = plt.get_cmap("turbo")
    for i in range(N_ACT):
        ax2.plot(np.arange(N_ENV_STEPS) * 8e-3, envs[:, i],
                 lw=1.4, color=act_cmap(i / (N_ACT - 1)), label=f"a{i}")
    ax2.set_xlim(0, N_ENV_STEPS * 8e-3)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("envelope")
    ax2.set_title(r"Envelope  $\sigma(a^{env}_i + c^{env}_i \cdot t/T)$  per actuator",
                  fontsize=10, weight="bold", loc="left")
    ax2.grid(True, alpha=0.3)

    # --- bottom: phi parameter groups as stacked bar ---
    ax3 = fig.add_subplot(gs[2])
    groups = [
        ("W",        40, "#4c72b0"),
        ("b",        10, "#dd8452"),
        ("g",        10, "#55a868"),
        (r"$a^{env}$", 10, "#c44e52"),
        (r"$c^{env}$", 10, "#8172b3"),
    ]
    offsets = np.cumsum([0] + [sz for _, sz, _ in groups[:-1]])
    for (lab, sz, color), off in zip(groups, offsets):
        seg = phi[off: off + sz]
        ax3.bar(np.arange(off, off + sz), seg, color=color, width=0.9)
        ax3.text(off + sz / 2, 0.55, lab, ha="center", va="bottom",
                 fontsize=10, color=color, weight="bold")
    ax3.set_ylim(-0.55, 0.55)
    ax3.set_xlim(-0.5, 80)
    ax3.axhline(0, color="0.5", lw=0.6)
    ax3.axhline(0.5, color="0.3", lw=0.5, ls="--")
    ax3.axhline(-0.5, color="0.3", lw=0.5, ls="--")
    ax3.set_xlabel("φ index")
    ax3.set_ylabel("value")
    ax3.set_title("Optimized φ* by parameter group  (dashed: ±0.5 bounds)",
                  fontsize=10, weight="bold", loc="left")

    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# 3. Cross-mode robustness table
# ---------------------------------------------------------------------------
def cross_mode_eval(x: np.ndarray, phi: np.ndarray, out_png: str, out_csv: str) -> None:
    cfg = MPMConfig(
        n_grid=64, voxel_dims=VOXEL_DIMS, scale=50.0,
        act_strength_base=24.0, shaping_weight=100.0, env_horizon=N_ENV_STEPS,
    )
    scene = build_scene(cfg)

    disps, returns = [], []
    for fr in FRICTION_MODES:
        r, d, _ = rollout_return(
            jnp.asarray(x), jnp.asarray(phi),
            jnp.asarray(fr, dtype=jnp.float32),
            scene, cfg, N_ENV_STEPS,
        )
        returns.append(float(r)); disps.append(float(d))

    print("  Cross-mode robustness:")
    print(f"    {'friction':>9} | {'disp':>8} | {'return':>8}")
    with open(out_csv, "w") as f:
        f.write("friction,disp,return\n")
        for fr, d, r in zip(FRICTION_MODES, disps, returns):
            print(f"    {fr:>9.2f} | {d:>8.5f} | {r:>8.3f}")
            f.write(f"{fr},{d},{r}\n")
    mean_d = float(np.mean(disps)); std_d = float(np.std(disps))
    print(f"    → mean disp = {mean_d:.5f}, std = {std_d:.5f}  "
          f"(CV={std_d/mean_d*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    colors = plt.get_cmap("tab10")(np.arange(len(FRICTION_MODES)))
    bars = ax.bar([f"{fr:.1f}" for fr in FRICTION_MODES], disps,
                  color=colors, edgecolor="white", linewidth=1.5)
    for b, d in zip(bars, disps):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                f"{d:.4f}", ha="center", fontsize=10, weight="bold")
    ax.axhline(mean_d, color="0.3", lw=1.2, ls="--",
               label=f"mean = {mean_d:.4f}  (CV={std_d/mean_d*100:.1f}%)")
    ax.set_xlabel("floor friction coefficient (mode)")
    ax.set_ylabel("final forward displacement")
    ax.set_title("Cross-mode robustness  —  main best θ* evaluated per mode",
                 fontsize=11, weight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_png}")
    print(f"  saved {out_csv}")


# ---------------------------------------------------------------------------
# 4. Fidelity FLOPs breakdown
# ---------------------------------------------------------------------------
def draw_fidelity_budget(bh: list, out: str) -> None:
    # Fidelity → env steps map (from adapters.py)
    FIDELITY_STEPS = {0: 30, 1: 100, 2: 200}
    # Assume M=32, num_modes=4 (from main config). Each diffusion step evaluates
    # M × num_modes rollouts at the step's fidelity level.
    M = 32
    num_modes = 4
    SUBSTEPS = 16  # substeps_per_env_step at frame_dt=8ms, dt=5e-4

    per_fid_evals = {0: 0, 1: 0, 2: 0}
    per_fid_diffsteps = {0: 0, 1: 0, 2: 0}
    for s in bh:
        fid = int(s["fidelity_level"])
        per_fid_diffsteps[fid] += 1
        per_fid_evals[fid] += M * num_modes
    # FLOPs proxy: evals × env_steps × substeps
    flops = {
        fid: per_fid_evals[fid] * FIDELITY_STEPS[fid] * SUBSTEPS
        for fid in [0, 1, 2]
    }
    total_flops = sum(flops.values())
    total_evals = sum(per_fid_evals.values())

    # Fine-only (hypothetical single-fidelity) for comparison
    total_diffsteps = sum(per_fid_diffsteps.values())
    fine_only_flops = total_diffsteps * M * num_modes * FIDELITY_STEPS[2] * SUBSTEPS

    print(f"  Fidelity budget breakdown  (M={M}, num_modes={num_modes}):")
    for fid in [0, 1, 2]:
        print(f"    fid={fid} ({FIDELITY_STEPS[fid]:>3} env-steps):  "
              f"{per_fid_diffsteps[fid]:>3} diffusion steps  "
              f"{per_fid_evals[fid]:>5} rollouts  "
              f"{flops[fid]/1e6:>7.2f} Msubsteps  ({flops[fid]/total_flops*100:>4.1f}%)")
    print(f"    ------------------------------------------------")
    print(f"    total multi-fidelity:  {total_evals} rollouts  "
          f"{total_flops/1e6:.2f} Msubsteps")
    print(f"    fine-only hypothetical: {total_diffsteps*M*num_modes} rollouts  "
          f"{fine_only_flops/1e6:.2f} Msubsteps  "
          f"({fine_only_flops/total_flops:.2f}x more)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=140)

    # --- left: stacked bar per fidelity ---
    ax = axes[0]
    fids = [0, 1, 2]
    colors = ["#eef5ff", "#cce6c9", "#ffd9a8"]
    borders = ["#5b8bd6", "#4e9559", "#d68b4a"]
    bottom = 0
    for fid in fids:
        f = flops[fid] / 1e6
        ax.bar(["multi-fidelity"], [f], bottom=bottom,
               color=colors[fid], edgecolor=borders[fid], linewidth=1.5,
               label=f"fid={fid}  ({FIDELITY_STEPS[fid]} steps)  "
                     f"{per_fid_diffsteps[fid]} diff-steps")
        ax.text(0, bottom + f / 2, f"{f:.2f} Msubsteps\n({flops[fid]/total_flops*100:.1f}%)",
                ha="center", va="center", fontsize=9, weight="bold")
        bottom += f
    # fine-only hypothetical next to it
    ax.bar(["fine-only (hypothetical)"], [fine_only_flops / 1e6],
           color="0.85", edgecolor="0.4", linewidth=1.5,
           label=f"fine-only, same # diff-steps ({total_diffsteps})")
    ax.text(1, fine_only_flops / 1e6 / 2,
            f"{fine_only_flops/1e6:.2f} Msubsteps\n"
            f"({fine_only_flops/total_flops:.2f}× multi-fid)",
            ha="center", va="center", fontsize=9, weight="bold", color="0.25")
    ax.set_ylabel("total rollout cost  (Msubsteps)")
    ax.set_title("Multi-fidelity vs fine-only budget",
                 fontsize=11, weight="bold", loc="left")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # --- right: diffusion-steps × fidelity stacked over k ---
    ax = axes[1]
    k_arr = np.array([s["k_forward"] for s in bh])
    fid_arr = np.array([s["fidelity_level"] for s in bh])
    env_arr = np.array([FIDELITY_STEPS[int(s["fidelity_level"])] for s in bh])
    for fid in fids:
        mask = fid_arr == fid
        ax.bar(k_arr[mask], env_arr[mask], width=1.0,
               color=colors[fid], edgecolor=borders[fid], linewidth=0.3,
               label=f"fid={fid}  ({FIDELITY_STEPS[fid]} env-steps)")
    ax.set_xlabel("diffusion step  k")
    ax.set_ylabel("env-steps per rollout")
    ax.set_title("Fidelity ladder schedule",
                 fontsize=11, weight="bold", loc="left")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
def main() -> None:
    x, phi, bh, final_return, wall_time = _load_result()
    print(f"Loaded main best θ*:  return={final_return:.3f}  wall_time={wall_time:.0f}s")
    print(f"  |x|={len(x)}, |phi|={len(phi)}, |bridge_history|={len(bh)}")
    print()

    print("[1/4] Morphology 3D render")
    draw_morphology(x, os.path.join(RESULTS_DIR, "morphology.png"))

    print("[2/4] Controller heatmap + envelope + phi bars")
    draw_controller(phi, os.path.join(RESULTS_DIR, "controller_heatmap.png"))

    print("[3/4] Cross-mode robustness (4 friction rollouts)")
    cross_mode_eval(
        x, phi,
        os.path.join(RESULTS_DIR, "cross_mode_robustness.png"),
        os.path.join(RESULTS_DIR, "cross_mode_robustness.csv"),
    )

    print("[4/4] Fidelity FLOPs budget breakdown")
    draw_fidelity_budget(bh, os.path.join(RESULTS_DIR, "fidelity_budget.png"))


if __name__ == "__main__":
    main()
