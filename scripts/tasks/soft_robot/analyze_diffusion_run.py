"""Parameterized post-hoc analysis for any diffusion-based co-design run.

For each result dir, produces:
  - morphology.png           : 3D voxel render (prior vs optimized)
  - controller_heatmap.png   : actuator × time activation + envelope + phi bars
  - cross_mode_robustness.{png,csv} : best θ evaluated at each friction mode
  - fidelity_budget.png      : FLOPs breakdown per fidelity
  - reward_vs_k.png          : env_return & R_s1 over diffusion steps
  - diffusion_evolution.png  : morphology + controller snapshots at sparse k

Reads physics knobs (act_strength_base, voxel_dims, backward_penalty_weight)
from the experiment's YAML so analysis matches training.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import yaml
import jax.numpy as jnp

from genedynamics.envs.external.jax_mpm.scene import (
    MPMConfig, build_scene, rollout_return, compute_actuation,
)
from genedynamics.envs.external.jax_mpm.adapters import FIDELITY_STEPS


# Hardcoded experiment-wide constants (keeps script simple; if any config
# ever changes these, add them to the per-run override dict below).
VOXEL_DIMS = (4, 3, 4)
N_ACT = 10
K_SIN = 4
N_ENV_STEPS = 200
FRICTION_MODES = [0.3, 0.4, 0.5, 0.6]


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _expand_x(x_opt: np.ndarray) -> np.ndarray:
    """Mirror optimizer x (24 half voxels) → 48 full voxels if z-symmetric."""
    if x_opt.shape[-1] == 48:
        return x_opt
    # Assume z-symmetric half-space: shape (24,) → (4, 3, 2) → (4, 3, 4)
    vx, vy, vz = VOXEL_DIMS
    half = x_opt.reshape(vx, vy, vz // 2)
    full = np.concatenate([half, half[:, :, ::-1]], axis=2)
    return full.reshape(-1)


def _make_cfg(act_strength: float, backward_pen: float) -> MPMConfig:
    return MPMConfig(
        n_grid=64, voxel_dims=VOXEL_DIMS, scale=50.0,
        act_strength_base=act_strength, shaping_weight=100.0,
        env_horizon=N_ENV_STEPS, backward_penalty_weight=backward_pen,
    )


# ---------------------------------------------------------------------------
# 1. Morphology 3D
# ---------------------------------------------------------------------------
def _render_cube(ax, grid_arr, title):
    """Render (vx, vy, vz) occupancy as 3D voxels with CORRECT orientation:
    sim_X (forward) → plot_X,  sim_Z (side) → plot_Y (depth),
    sim_Y (up)      → plot_Z   so matplotlib's vertical axis matches gravity.
    """
    vx, vy, vz = VOXEL_DIMS
    cmap = plt.get_cmap("viridis")
    gap = 0.08

    def cube_verts(cx, cy, cz, s=0.92):
        h = s / 2; d = [-h, h]
        return np.array([[cx+dx, cy+dy, cz+dz] for dx in d for dy in d for dz in d])
    faces_idx = [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
    for i in range(vx):          # sim_X index
        for j in range(vy):       # sim_Y index (up)
            for k in range(vz):   # sim_Z index (side)
                # Remap: sim_Y → plot_Z (vertical), sim_Z → plot_Y (depth)
                cx = i*(1+gap); cy = k*(1+gap); cz = j*(1+gap)
                v = cube_verts(cx, cy, cz)
                occ = float(grid_arr[i, j, k])
                alpha = float(np.clip(0.25 + 0.70 * (occ - 0.2) / 0.8, 0.1, 0.95))
                faces = [[v[p] for p in face] for face in faces_idx]
                ax.add_collection3d(Poly3DCollection(
                    faces, facecolors=cmap(occ), edgecolors="0.3",
                    linewidths=0.4, alpha=alpha,
                ))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel("X (fwd)", labelpad=-5, fontsize=8)
    ax.set_ylabel("Z (side)", labelpad=-5, fontsize=8)
    ax.set_zlabel("Y (up)",  labelpad=-5, fontsize=8)
    ax.view_init(elev=22, azim=-60)
    ax.set_title(title, fontsize=11, weight="bold")


def draw_morphology(x_full: np.ndarray, out: str, label: str) -> None:
    vx, vy, vz = VOXEL_DIMS
    grid = x_full.reshape(vx, vy, vz)
    fig = plt.figure(figsize=(12, 5), dpi=140)
    ax1 = fig.add_subplot(121, projection="3d")
    _render_cube(ax1, np.full(grid.shape, 0.6, dtype=np.float32), "prior (uniform 0.6)")
    ax2 = fig.add_subplot(122, projection="3d")
    _render_cube(ax2, grid, "optimized  x*")
    cmap = plt.get_cmap("viridis")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.2, 1.0))
    cax = fig.add_axes([0.93, 0.25, 0.015, 0.5])
    cb = plt.colorbar(sm, cax=cax); cb.set_label("voxel occupancy")
    stats = (f"occupancy min={grid.min():.2f} mean={grid.mean():.2f} max={grid.max():.2f}   "
             f"min-bound: {int(np.isclose(grid, 0.2, atol=0.02).sum())}/{grid.size}   "
             f"max-bound: {int(np.isclose(grid, 1.0, atol=0.02).sum())}/{grid.size}")
    fig.text(0.5, 0.02, stats, ha="center", fontsize=9, color="0.25", family="monospace")
    fig.suptitle(f"Morphology  x  —  {label}", fontsize=13, weight="bold")
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Controller heatmap
# ---------------------------------------------------------------------------
def draw_controller(phi: np.ndarray, out: str, label: str) -> None:
    cfg = MPMConfig(n_grid=64, voxel_dims=VOXEL_DIMS, env_horizon=N_ENV_STEPS)
    act_mat = np.zeros((N_ACT, N_ENV_STEPS), dtype=np.float32)
    for t in range(N_ENV_STEPS):
        a = compute_actuation(jnp.asarray(phi, dtype=jnp.float32),
                              jnp.asarray(t, dtype=jnp.int32), cfg,
                              v_com_x=jnp.asarray(0.0, dtype=jnp.float32))
        act_mat[:, t] = np.asarray(a)

    fig = plt.figure(figsize=(11, 6), dpi=140)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.0, 1.0], hspace=0.55)

    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(act_mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                    origin="lower", extent=[0, N_ENV_STEPS*8e-3, -0.5, N_ACT-0.5])
    ax1.set_yticks(range(N_ACT)); ax1.set_ylabel("actuator\n(0=back · 9=front)")
    ax1.set_xlabel("time (s)")
    ax1.set_title(r"Actuator output  $a_i(t)$", fontsize=11, weight="bold", loc="left")
    plt.colorbar(im, ax=ax1, fraction=0.035, pad=0.01)

    ax2 = fig.add_subplot(gs[1])
    a_env = phi[60:70]; c_env = phi[70:80]
    t_frac = np.arange(N_ENV_STEPS) / N_ENV_STEPS
    envs = 1.0 / (1.0 + np.exp(-(a_env[None, :] + c_env[None, :] * t_frac[:, None])))
    act_cmap = plt.get_cmap("turbo")
    for i in range(N_ACT):
        ax2.plot(np.arange(N_ENV_STEPS) * 8e-3, envs[:, i], lw=1.4,
                 color=act_cmap(i / (N_ACT - 1)))
    ax2.set_xlim(0, N_ENV_STEPS*8e-3); ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("envelope")
    ax2.set_title(r"Envelope  $\sigma(a^{env}_i + c^{env}_i t/T)$",
                  fontsize=10, weight="bold", loc="left")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[2])
    groups = [("W", 40, "#4c72b0"), ("b", 10, "#dd8452"), ("g", 10, "#55a868"),
              (r"$a^{env}$", 10, "#c44e52"), (r"$c^{env}$", 10, "#8172b3")]
    offsets = np.cumsum([0] + [sz for _, sz, _ in groups[:-1]])
    for (lab, sz, color), off in zip(groups, offsets):
        seg = phi[off:off+sz]
        ax3.bar(np.arange(off, off+sz), seg, color=color, width=0.9)
        ax3.text(off+sz/2, 0.55, lab, ha="center", va="bottom", fontsize=10,
                 color=color, weight="bold")
    ax3.set_ylim(-0.55, 0.55); ax3.set_xlim(-0.5, 80); ax3.axhline(0, color="0.5", lw=0.6)
    for y in (0.5, -0.5): ax3.axhline(y, color="0.3", lw=0.5, ls="--")
    ax3.set_xlabel("φ index"); ax3.set_ylabel("value")
    ax3.set_title("φ* by group", fontsize=10, weight="bold", loc="left")
    fig.suptitle(label, fontsize=12, weight="bold", y=1.00)
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Cross-mode robustness
# ---------------------------------------------------------------------------
def cross_mode_eval(x_full: np.ndarray, phi: np.ndarray, act_strength: float,
                    backward_pen: float, out_png: str, out_csv: str,
                    label: str) -> dict:
    cfg = _make_cfg(act_strength, backward_pen)
    scene = build_scene(cfg)
    disps, returns = [], []
    for fr in FRICTION_MODES:
        r, d, _ = rollout_return(jnp.asarray(x_full), jnp.asarray(phi),
                                 jnp.asarray(fr, dtype=jnp.float32),
                                 scene, cfg, N_ENV_STEPS)
        returns.append(float(r)); disps.append(float(d))
    disps = np.asarray(disps); returns = np.asarray(returns)
    mean, std = disps.mean(), disps.std()
    cv = std / (mean + 1e-8) * 100

    with open(out_csv, "w") as f:
        f.write("friction,disp,return\n")
        for fr, d, r in zip(FRICTION_MODES, disps, returns):
            f.write(f"{fr},{d},{r}\n")

    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    colors = plt.get_cmap("tab10")(np.arange(len(FRICTION_MODES)))
    bars = ax.bar([f"{fr:.1f}" for fr in FRICTION_MODES], disps,
                  color=colors, edgecolor="white", linewidth=1.5)
    for b, d in zip(bars, disps):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002, f"{d:.4f}",
                ha="center", fontsize=10, weight="bold")
    ax.axhline(mean, color="0.3", lw=1.2, ls="--",
               label=f"mean={mean:.4f}  CV={cv:.1f}%  min/max={disps.min()/disps.max():.2f}")
    ax.set_xlabel("friction"); ax.set_ylabel("final disp")
    ax.set_title(f"Cross-mode robustness  —  {label}", fontsize=11, weight="bold")
    ax.grid(True, alpha=0.3, axis="y"); ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(out_png, dpi=160, bbox_inches="tight"); plt.close(fig)
    return {"disps": disps.tolist(), "mean": float(mean), "std": float(std),
            "cv": float(cv), "min_over_max": float(disps.min()/disps.max())}


# ---------------------------------------------------------------------------
# 4. Fidelity budget
# ---------------------------------------------------------------------------
def draw_fidelity_budget(bh: list, M: int, num_modes: int, out: str,
                         label: str) -> None:
    SUBSTEPS = 16
    per_fid_d = {fid: 0 for fid in FIDELITY_STEPS}
    per_fid_e = {fid: 0 for fid in FIDELITY_STEPS}
    for s in bh:
        fid = int(s["fidelity_level"])
        per_fid_d[fid] = per_fid_d.get(fid, 0) + 1
        per_fid_e[fid] = per_fid_e.get(fid, 0) + M * num_modes
    flops = {fid: per_fid_e[fid] * FIDELITY_STEPS[fid] * SUBSTEPS
             for fid in FIDELITY_STEPS}
    total_flops = max(sum(flops.values()), 1)
    total_d = sum(per_fid_d.values())
    # Hypothetical fine-only
    fine_steps = FIDELITY_STEPS[max(FIDELITY_STEPS)]
    fine_only_flops = total_d * M * num_modes * fine_steps * SUBSTEPS

    fids = sorted(FIDELITY_STEPS)
    colors = ["#eef5ff", "#cce6c9", "#ffd9a8"]
    borders = ["#5b8bd6", "#4e9559", "#d68b4a"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=140)

    ax = axes[0]
    bottom = 0
    for i, fid in enumerate(fids):
        f = flops[fid] / 1e6
        ax.bar(["multi-fidelity"], [f], bottom=bottom,
               color=colors[i], edgecolor=borders[i], linewidth=1.5,
               label=f"fid={fid}  ({FIDELITY_STEPS[fid]}-steps) · {per_fid_d[fid]} diff-steps")
        if f > 0.05:
            ax.text(0, bottom+f/2, f"{f:.2f}M\n({flops[fid]/total_flops*100:.1f}%)",
                    ha="center", va="center", fontsize=9, weight="bold")
        bottom += f
    ax.bar(["fine-only (hypothetical)"], [fine_only_flops/1e6],
           color="0.85", edgecolor="0.4", linewidth=1.5,
           label=f"all-fine, same #diff-steps")
    ax.text(1, fine_only_flops/1e6/2,
            f"{fine_only_flops/1e6:.2f}M\n({fine_only_flops/total_flops:.2f}× multi-fid)",
            ha="center", va="center", fontsize=9, weight="bold", color="0.25")
    ax.set_ylabel("rollout cost  (Msubsteps)")
    ax.set_title("Multi-fidelity vs fine-only", fontsize=11, weight="bold", loc="left")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    k_arr = np.array([s["k_forward"] for s in bh])
    fid_arr = np.array([s["fidelity_level"] for s in bh])
    env_arr = np.array([FIDELITY_STEPS[int(s["fidelity_level"])] for s in bh])
    for i, fid in enumerate(fids):
        mask = fid_arr == fid
        if mask.sum() > 0:
            ax.bar(k_arr[mask], env_arr[mask], width=1.0,
                   color=colors[i], edgecolor=borders[i], linewidth=0.3,
                   label=f"fid={fid}")
    ax.set_xlabel("diffusion step  k")
    ax.set_ylabel("env-steps per rollout")
    ax.set_title("Fidelity schedule", fontsize=11, weight="bold", loc="left")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(label, fontsize=12, weight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# 5. reward vs k
# ---------------------------------------------------------------------------
def draw_reward_vs_k(bh: list, out: str, label: str) -> None:
    k = np.array([s["k_forward"] for s in bh])
    env = np.array([s["mean_env_return"] for s in bh])
    rs1 = np.array([s["mean_R_s1"] for s in bh])
    fid = np.array([s["fidelity_level"] for s in bh])

    fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
    fid_colors = {0: "#eef5ff", 1: "#e6fff0", 2: "#fff5e6"}
    blocks = []
    if len(fid) > 0:
        cur_fid = int(fid[0]); cur_start = 0
        for i in range(1, len(fid)):
            if int(fid[i]) != cur_fid:
                blocks.append((cur_fid, cur_start, i))
                cur_fid = int(fid[i]); cur_start = i
        blocks.append((cur_fid, cur_start, len(fid)))
        for f, s, e in blocks:
            ax.axvspan(k[s]-0.5, k[e-1]+0.5, color=fid_colors.get(f, "#eee"),
                       alpha=0.7, zorder=0)

    ax.plot(k, env, "-o", lw=1.5, ms=3, color="#d62728", label="mean env return")
    ax.plot(k, rs1, "-", lw=1.2, color="#1f77b4", alpha=0.7, label="mean $R_{s1}$")
    ax.set_xlabel("diffusion step  k"); ax.set_ylabel("reward")
    ax.set_title(f"Reward vs k  —  {label}", fontsize=11, weight="bold")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper left", fontsize=9)

    y_top = ax.get_ylim()[1]
    for f, s, e in blocks:
        ax.text((k[s]+k[e-1])/2, y_top*0.98,
                f"fid={f}  ({FIDELITY_STEPS.get(f, '?')} steps)",
                ha="center", va="top", fontsize=9, color="0.3",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))
    fig.tight_layout(); fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Diffusion evolution (NEW — uses theta_history)
# ---------------------------------------------------------------------------
def _draw_3d_voxels(ax, grid_arr, elev=22, azim=-60, gap=0.08, title=None):
    """Render (vx, vy, vz) occupancy as 3D voxels on the given Axes3D.
    Remap sim axes → plot axes: sim_X→plot_X, sim_Z→plot_Y, sim_Y→plot_Z,
    so matplotlib's vertical axis matches gravity (Y is up in sim coords)."""
    vx, vy, vz = grid_arr.shape
    cmap = plt.get_cmap("viridis")
    faces_idx = [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
    for i in range(vx):          # sim_X  (forward)
        for j in range(vy):       # sim_Y  (up)
            for k in range(vz):   # sim_Z  (side)
                cx = i*(1+gap); cy = k*(1+gap); cz = j*(1+gap)
                h = 0.46
                v = np.array([[cx+dx, cy+dy, cz+dz]
                              for dx in (-h, h) for dy in (-h, h) for dz in (-h, h)])
                occ = float(grid_arr[i, j, k])
                alpha = float(np.clip(0.20 + 0.70 * (occ - 0.2) / 0.8, 0.10, 0.95))
                faces = [[v[p] for p in face] for face in faces_idx]
                ax.add_collection3d(Poly3DCollection(
                    faces, facecolors=cmap(occ), edgecolors="0.3",
                    linewidths=0.3, alpha=alpha,
                ))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel("X (fwd)", labelpad=-8, fontsize=7)
    ax.set_ylabel("Z (side)", labelpad=-8, fontsize=7)
    ax.set_zlabel("Y (up)",  labelpad=-8, fontsize=7)
    ax.view_init(elev=elev, azim=azim)
    span = max(vx, vy, vz) * (1 + gap)
    ax.set_xlim(-0.5, span); ax.set_ylim(-0.5, span); ax.set_zlim(-0.5, span)
    if title is not None:
        ax.set_title(title, fontsize=9, weight="bold")


def draw_diffusion_evolution(theta_history: np.ndarray, bh: list, x_dim: int,
                             out: str, label: str) -> None:
    """Snapshot morphology + controller activation along the REVERSE diffusion.

    theta_history[i] is the denoised θ after the i-th reverse step (i=0 is the
    first step from noise, i=K-1 is the final clean solution). We label panels
    with k_reverse_idx = K-1-i, so they read left→right as k=K-1 (noise) down
    to k=0 (clean), matching standard diffusion notation.
    """
    K = theta_history.shape[0]
    if K == 0:
        print(f"    (skipping diffusion_evolution — no theta_history)")
        return
    k_reverse_targets = [K-1, int(0.8*(K-1)), int(0.6*(K-1)),
                         int(0.4*(K-1)), int(0.2*(K-1)), 0]
    k_reverse_targets = sorted(set(k_reverse_targets), reverse=True)
    panels = [(K - 1 - kr, kr) for kr in k_reverse_targets]
    n_cols = len(panels)

    vx, vy, vz = VOXEL_DIMS

    fig = plt.figure(figsize=(2.5*n_cols + 1.2, 7.5), dpi=140)
    gs = fig.add_gridspec(2, n_cols, height_ratios=[1.2, 0.9], hspace=0.25,
                          wspace=0.12)

    for col, (i_iter, k_rev) in enumerate(panels):
        theta_k = theta_history[i_iter]
        x_opt = theta_k[:x_dim]
        phi_k = theta_k[x_dim:]
        x_full = _expand_x(np.asarray(x_opt))
        grid = x_full.reshape(vx, vy, vz)

        # Top: 3D morphology
        ax_m = fig.add_subplot(gs[0, col], projection="3d")
        env_ret = bh[i_iter].get("mean_env_return", float("nan")) if i_iter < len(bh) else float("nan")
        tag = " (noise)" if k_rev == K-1 else (" (clean)" if k_rev == 0 else "")
        _draw_3d_voxels(ax_m, grid, title=f"k={k_rev}{tag}\nenv_return={env_ret:.2f}")

        # Bottom: controller heatmap
        ax_c = fig.add_subplot(gs[1, col])
        act_mat = _phi_to_act_mat(np.asarray(phi_k))
        ax_c.imshow(act_mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                    origin="lower")
        ax_c.set_xticks([]); ax_c.set_yticks([])
        if col == 0:
            ax_c.set_ylabel("actuator", fontsize=9)

    fig.text(0.005, 0.70, "morphology\n(3D voxels)", ha="left", va="center",
             fontsize=10, weight="bold", rotation=90)
    fig.text(0.005, 0.27, "actuator\nactivity", ha="left", va="center",
             fontsize=10, weight="bold", rotation=90)
    fig.text(0.5, 0.97,
             "reverse diffusion:   k = K−1 (noise)  →  k = 0 (clean θ*)",
             ha="center", va="top", fontsize=11, weight="bold",
             color="#d62728")
    fig.suptitle(f"Diffusion evolution of θ*  —  {label}",
                 fontsize=12, weight="bold", y=1.02)
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Diffusion evolution GIF (morph + controller, one frame per k)
# ---------------------------------------------------------------------------
def _phi_to_act_mat(phi: np.ndarray, omega: float = 20.0,
                    K: int = 4, n_act: int = 10,
                    num_env_steps: int = 200, frame_dt: float = 8e-3
                    ) -> np.ndarray:
    """Vectorized numpy version of compute_actuation at v_com_x=0."""
    W = phi[:n_act*K].reshape(n_act, K)
    b = phi[n_act*K:n_act*K + n_act]
    a_env = phi[n_act*K + 2*n_act : n_act*K + 3*n_act]
    c_env = phi[n_act*K + 3*n_act : n_act*K + 4*n_act]
    t = np.arange(num_env_steps) * frame_dt                      # (T,)
    phases = 2*np.pi/K * np.arange(K)                             # (K,)
    basis = np.sin(omega * t[:, None] + phases[None, :])          # (T, K)
    raw = np.tanh(basis @ W.T + b[None, :])                       # (T, n_act)
    t_frac = np.arange(num_env_steps) / num_env_steps
    env = 1.0 / (1.0 + np.exp(-(a_env[None, :] + c_env[None, :] * t_frac[:, None])))
    return (raw * env).T                                          # (n_act, T)


def render_diffusion_gif(theta_history: np.ndarray, bh: list, x_dim: int,
                         out: str, label: str) -> None:
    """One GIF: per k (reverse order K-1→0) one frame showing morph + controller.

    Uses numpy compute_actuation (no JIT overhead) so 100 frames render in ~10s.
    """
    import imageio.v3 as iio
    K = theta_history.shape[0]
    if K == 0:
        return
    vx, vy, vz = VOXEL_DIMS

    # Pre-compute global vmin/vmax for morphology so the colorbar is stable.
    # Per-frame it's just [0.2, 1.0].

    # Preflight: compute disp scale of env_return series for frame header.
    env_returns = [s.get("mean_env_return", 0.0) for s in bh]

    tmp_frames = []
    tmp_dir = os.path.dirname(out)
    frame_dir = os.path.join(tmp_dir, "_diff_gif_frames")
    os.makedirs(frame_dir, exist_ok=True)

    for i_iter in range(K):
        k_rev = K - 1 - i_iter
        theta_k = theta_history[i_iter]
        x_full = _expand_x(np.asarray(theta_k[:x_dim]))
        grid = x_full.reshape(vx, vy, vz)
        phi_k = np.asarray(theta_k[x_dim:])
        act_mat = _phi_to_act_mat(phi_k)  # (n_act, T)

        fig = plt.figure(figsize=(9.2, 3.6), dpi=110)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.6], wspace=0.25)

        # Morph: 3D voxel render (side angle)
        ax_m = fig.add_subplot(gs[0], projection="3d")
        _draw_3d_voxels(ax_m, grid, title="morphology (3D)")

        # Controller: actuator × time
        ax_c = fig.add_subplot(gs[1])
        ax_c.imshow(act_mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                    origin="lower", extent=[0, N_ENV_STEPS*8e-3, -0.5, N_ACT-0.5])
        ax_c.set_yticks([0, 4, 9])
        ax_c.set_ylabel("actuator", fontsize=8)
        ax_c.set_xlabel("time (s)", fontsize=8)
        ax_c.set_title(r"$a_i(t)$", fontsize=9, weight="bold")

        tag = "  (noise)" if k_rev == K-1 else ("  (clean)" if k_rev == 0 else "")
        env_ret = env_returns[i_iter] if i_iter < len(env_returns) else float("nan")
        fig.suptitle(
            f"{label}   k = {k_rev}{tag}   env_return = {env_ret:.2f}",
            fontsize=10, weight="bold", y=1.04,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fpath = os.path.join(frame_dir, f"f_{i_iter:04d}.png")
        fig.savefig(fpath, dpi=110, bbox_inches="tight")
        plt.close(fig)
        tmp_frames.append(fpath)

    # Compose GIF — reverse frames so playback goes noise → clean.
    frames = [iio.imread(p) for p in tmp_frames]
    iio.imwrite(out, frames, duration=80, loop=0)
    for p in tmp_frames:
        try: os.remove(p)
        except OSError: pass
    try: os.rmdir(frame_dir)
    except OSError: pass


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def analyze(config_path: str) -> None:
    cfg_data = _load_config(config_path)
    out_dir = cfg_data["output_dir"]
    if not os.path.isabs(out_dir):
        out_dir = os.path.join("/workspace/genedynamics", out_dir)
    results_json = os.path.join(out_dir, "results.json")
    if not os.path.exists(results_json):
        print(f"[skip] {config_path}: no results.json at {out_dir}")
        return

    label = cfg_data.get("name", os.path.basename(out_dir))
    eval_params = cfg_data.get("evaluator_runtime", {}) or {}
    act_strength = float(eval_params.get("act_strength_base", 24.0))
    backward_pen = float(eval_params.get("backward_penalty_weight", 0.0))

    method_params = cfg_data.get("method_params", {}) or {}
    num_modes = int(method_params.get("num_modes", 4))
    scheduler = cfg_data.get("scheduler_config", {}) or {}
    diff = (scheduler.get("diffusion_schedulers") or [{}])[0]
    M = int(diff.get("M_k", 32))

    with open(results_json) as f:
        data = json.load(f)
    r = data[0]["result"]
    x = np.asarray(r["x"], dtype=np.float32)
    phi = np.asarray(r["phi"], dtype=np.float32)
    bh = r.get("bridge_history", [])
    theta_hist = np.asarray(r.get("theta_history", []), dtype=np.float32)

    x_full = _expand_x(x) if x.shape[-1] != 48 else x
    x_dim_opt = 24 if x.shape[-1] == 24 else (x.shape[-1])  # fallback

    print(f"  [{label}]  return={r.get('return_', 0):.3f}  "
          f"disp(fr=0.45)... (running cross-mode next)")

    os.makedirs(out_dir, exist_ok=True)
    draw_morphology(x_full, os.path.join(out_dir, "morphology.png"), label)
    draw_controller(phi, os.path.join(out_dir, "controller_heatmap.png"), label)
    cm = cross_mode_eval(x_full, phi, act_strength, backward_pen,
                         os.path.join(out_dir, "cross_mode_robustness.png"),
                         os.path.join(out_dir, "cross_mode_robustness.csv"),
                         label)
    print(f"    cross-mode: mean={cm['mean']:.4f}  CV={cm['cv']:.1f}%  "
          f"min/max={cm['min_over_max']:.2f}")
    draw_fidelity_budget(bh, M, num_modes,
                         os.path.join(out_dir, "fidelity_budget.png"), label)
    draw_reward_vs_k(bh, os.path.join(out_dir, "reward_vs_k.png"), label)
    if theta_hist.size > 0:
        # Derive x_dim from theta_hist: total D - phi_dim(80)
        x_dim_opt = theta_hist.shape[1] - 80
        draw_diffusion_evolution(theta_hist, bh, x_dim_opt,
                                 os.path.join(out_dir, "diffusion_evolution.png"),
                                 label)
        print(f"    diffusion_evolution.png: K={theta_hist.shape[0]} snapshots")
        render_diffusion_gif(theta_hist, bh, x_dim_opt,
                             os.path.join(out_dir, "diffusion_evolution.gif"),
                             label)
        print(f"    diffusion_evolution.gif: 100 frames")
    else:
        print(f"    diffusion_evolution: SKIPPED (theta_history missing)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+",
                        help="YAML configs to analyze (must have output_dir + results.json)")
    args = parser.parse_args()
    for cfg in args.configs:
        print(f"\n=== {cfg} ===")
        analyze(cfg)


if __name__ == "__main__":
    main()
