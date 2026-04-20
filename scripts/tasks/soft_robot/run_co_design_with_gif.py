#!/usr/bin/env python3
"""
Run co-design experiment from YAML config, then render best-theta rollout as GIF.

Wraps run_co_design.py; after planning, rebuilds env, rolls out best (x*, phi*)
while capturing particle positions, and renders a top-down/side matplotlib GIF.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml


def _render_voxel_gif(
    vox_centroids: np.ndarray,        # (T, n_voxels, 3) in sim coords
    vox_occupancy: np.ndarray,        # (n_voxels,)
    voxel_dims: tuple,                # (vx, vy, vz)
    out_path: Path,
    return_value: float,
    title: str,
) -> None:
    """Voxel-level rollout replay: each of n_voxels is drawn as a small cube at
    its moving centroid, colored by its static occupancy (same colormap as
    morphology.png). Matches the morphology visualization — the cube mesh is
    the same lattice the optimizer sees. Semi-transparent tan floor grid.

    Coordinate remap: sim_X→plot_X (fwd), sim_Z→plot_Y (depth), sim_Y→plot_Z (up).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import imageio.v3 as iio

    T, n_voxels, _ = vox_centroids.shape
    vx, vy, vz = voxel_dims

    # Remap to plot coords
    cx_all = vox_centroids[..., 0]
    cy_all = vox_centroids[..., 2]     # plot_Y from sim_Z
    cz_all = vox_centroids[..., 1]     # plot_Z from sim_Y

    com = vox_centroids.mean(axis=1)
    com_px = com[:, 0]; com_py = com[:, 2]; com_pz = com[:, 1]

    x_min = float(cx_all.min() - 0.03); x_max = float(cx_all.max() + 0.03)
    y_min = float(cy_all.min() - 0.02); y_max = float(cy_all.max() + 0.02)
    z_floor = float(cz_all.min() - 0.012)   # floor just below the body
    z_top   = float(cz_all.max() + 0.03)

    cmap = plt.get_cmap("viridis")
    # Voxel cube half-size. Body is 0.10×0.06×0.10 split over 4×3×4 voxels, so
    # the *physical* voxel is 0.025×0.020×0.025 (not cubic because Y pitch is
    # smaller). For visual clarity we render all voxels as equal-sized cubes
    # using the smallest pitch as the half-extent.
    pitch_x = (cx_all.max() - cx_all.min()) / max(vx - 1, 1)
    pitch_y = (cy_all.max() - cy_all.min()) / max(vz - 1, 1)   # sim_Z pitch
    pitch_z = (cz_all.max() - cz_all.min()) / max(vy - 1, 1)   # sim_Y pitch
    h_cube = min(pitch_x, pitch_y, pitch_z) * 0.42
    hx = hy = hz = h_cube

    faces_idx = [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]

    def _draw_floor(ax):
        xs = np.linspace(x_min, x_max, 2)
        ys = np.linspace(y_min, y_max, 2)
        X, Y = np.meshgrid(xs, ys)
        Z = np.full_like(X, z_floor)
        ax.plot_surface(X, Y, Z, color="#c9b89a", alpha=0.30, linewidth=0,
                        antialiased=True)
        for gx in np.arange(round(x_min*50)/50, x_max+1e-6, 0.02):
            ax.plot([gx, gx], [y_min, y_max], [z_floor, z_floor],
                    color="0.55", lw=0.4, alpha=0.6)
        for gy in np.arange(round(y_min*50)/50, y_max+1e-6, 0.02):
            ax.plot([x_min, x_max], [gy, gy], [z_floor, z_floor],
                    color="0.55", lw=0.4, alpha=0.6)

    def _draw_voxels(ax, t_idx):
        # Draw all voxels with alpha gated on occupancy (ghost voxels hidden).
        for v in range(n_voxels):
            occ = float(vox_occupancy[v])
            if occ < 0.15:
                continue  # ghost voxel
            cx = float(cx_all[t_idx, v])
            cy = float(cy_all[t_idx, v])
            cz = float(cz_all[t_idx, v])
            verts = np.array([[cx + dx, cy + dy, cz + dz]
                              for dx in (-hx, hx)
                              for dy in (-hy, hy)
                              for dz in (-hz, hz)])
            faces = [[verts[p] for p in f] for f in faces_idx]
            alpha = float(np.clip(0.40 + 0.55 * (occ - 0.2) / 0.8, 0.2, 0.95))
            ax.add_collection3d(Poly3DCollection(
                faces, facecolors=cmap(occ), edgecolors="0.25",
                linewidths=0.5, alpha=alpha,
            ))

    tmp_dir = out_path.parent / f"_frames_{out_path.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    stride = max(1, T // 100)
    frame_paths = []

    for t_idx in range(0, T, stride):
        fig = plt.figure(figsize=(8.5, 4.5), dpi=100)
        ax = fig.add_subplot(projection="3d")
        ax.set_proj_type("ortho")
        _draw_floor(ax)
        _draw_voxels(ax, t_idx)
        # Prominent COM trail (thick red) + floor shadow (orange dashed)
        if t_idx > 0:
            ax.plot(com_px[: t_idx + 1], com_py[: t_idx + 1], com_pz[: t_idx + 1],
                    color="#cc0000", lw=3.0, alpha=0.95, zorder=10)
            ax.plot(com_px[: t_idx + 1], com_py[: t_idx + 1],
                    np.full(t_idx + 1, z_floor),
                    color="#ff9500", lw=1.8, ls="--", alpha=0.9, zorder=1)
        ax.scatter([com_px[t_idx]], [com_py[t_idx]], [com_pz[t_idx]],
                   s=80, c="#cc0000", marker="o", edgecolors="white",
                   lw=1.6, zorder=11)

        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_floor, z_top)
        try:
            ax.set_box_aspect((x_max - x_min, y_max - y_min, z_top - z_floor))
        except Exception:
            pass
        ax.set_xlabel("X  (forward)", fontsize=9, labelpad=2)
        ax.set_ylabel("Z  (side)",    fontsize=9, labelpad=2)
        ax.set_zlabel("Y  (up)",      fontsize=9, labelpad=2)
        ax.tick_params(axis="both", labelsize=7, pad=-1)
        ax.view_init(elev=22, azim=-60)
        forward = com_px[t_idx] - com_px[0]
        ax.set_title(
            f"{title}\nstep={t_idx}/{T}   return={return_value:.3f}   "
            f"forward=+{forward:.3f}",
            fontsize=10,
        )
        fpath = tmp_dir / f"f_{t_idx:04d}.png"
        fig.savefig(fpath, bbox_inches="tight"); plt.close(fig)
        frame_paths.append(fpath)

    frames = [iio.imread(p) for p in frame_paths]
    iio.imwrite(out_path, frames, duration=60, loop=0)
    for p in frame_paths: p.unlink(missing_ok=True)
    try: tmp_dir.rmdir()
    except OSError: pass

    # Static summary: voxel-style start/end + COM trail
    summary_path = out_path.with_suffix(".summary.png")
    fig = plt.figure(figsize=(13, 4.5), dpi=110)
    for i, (t_idx, lab) in enumerate(zip([0, T - 1], ["start", "end"])):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.set_proj_type("ortho")
        _draw_floor(ax)
        _draw_voxels(ax, t_idx)
        ax.plot(com_px[: t_idx + 1], com_py[: t_idx + 1], com_pz[: t_idx + 1],
                color="#cc0000", lw=3.0, alpha=0.95)
        ax.plot(com_px[: t_idx + 1], com_py[: t_idx + 1],
                np.full(t_idx + 1, z_floor),
                color="#ff9500", lw=1.6, ls="--")
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_floor, z_top)
        try:
            ax.set_box_aspect((x_max - x_min, y_max - y_min, z_top - z_floor))
        except Exception:
            pass
        ax.tick_params(axis="both", labelsize=7)
        ax.set_xlabel("X  (forward)", fontsize=9)
        ax.set_ylabel("Z  (side)",    fontsize=9)
        ax.set_zlabel("Y  (up)",      fontsize=9)
        ax.view_init(elev=22, azim=-60)
        ax.set_title(f"{lab}  step={t_idx}", fontsize=11, weight="bold")
    fig.suptitle(
        f"{title}   return={return_value:.3f}   "
        f"forward=+{com_px[-1]-com_px[0]:.3f}",
        fontsize=12, weight="bold",
    )
    fig.tight_layout(); fig.savefig(summary_path, bbox_inches="tight"); plt.close(fig)


def _render_gif(positions_over_time: np.ndarray, out_path: Path, return_value: float, title: str) -> None:
    """
    positions_over_time: (T, N, 3) particle positions in sim coords (X, Y, Z)
    where Y is vertical (gravity acts on [..., 1]).

    Renders via coordinate remap: sim_Y → plot_Z (vertical), sim_Z → plot_Y
    (depth), so matplotlib's built-in Z-is-up convention matches the physics.
    Fixed camera, orthographic, explicit "+X forward" arrow on the floor.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v3 as iio

    pos = positions_over_time  # (T, N, 3) in sim coords (x, y, z)
    T, N, _ = pos.shape

    # Remap to plot coords: plot_X = sim_X, plot_Y = sim_Z (depth), plot_Z = sim_Y (up).
    px = pos[..., 0]      # forward
    py = pos[..., 2]      # side/depth
    pz = pos[..., 1]      # up

    init_x = px[0]
    rmin, rmax = float(init_x.min()), float(init_x.max())
    norm_x = (init_x - rmin) / max(rmax - rmin, 1e-6)
    colors = plt.get_cmap("turbo")(norm_x)

    com_x = px.mean(axis=1); com_y = py.mean(axis=1); com_z = pz.mean(axis=1)

    x_min = float(min(px.min(), com_x.min()) - 0.03)
    x_max = float(max(px.max(), com_x.max()) + 0.03)
    y_min = float(py.min() - 0.02)
    y_max = float(py.max() + 0.02)
    z_floor = float(pz.min())
    z_top   = float(pz.max() + 0.05)

    def _draw_floor_and_arrow(ax):
        # Semi-transparent tan floor so the body is never visually occluded.
        xs = np.linspace(x_min, x_max, 2)
        ys = np.linspace(y_min, y_max, 2)
        X, Y = np.meshgrid(xs, ys)
        Z = np.full_like(X, z_floor)
        ax.plot_surface(X, Y, Z, color="#c9b89a", alpha=0.35, linewidth=0,
                        antialiased=True)
        # Dense grid lines on the floor for depth cue (every 0.02 m)
        for gx in np.arange(round(x_min*50)/50, x_max+1e-6, 0.02):
            ax.plot([gx, gx], [y_min, y_max], [z_floor, z_floor],
                    color="0.55", lw=0.4, alpha=0.6)
        for gy in np.arange(round(y_min*50)/50, y_max+1e-6, 0.02):
            ax.plot([x_min, x_max], [gy, gy], [z_floor, z_floor],
                    color="0.55", lw=0.4, alpha=0.6)
        # Forward arrow along +X, on the floor, at the body's z-center.
        y_mid = (y_min + y_max) / 2
        ax.quiver(x_min + 0.015, y_mid, z_floor + 0.002,
                  (x_max - x_min) * 0.22, 0, 0,
                  color="#d62728", arrow_length_ratio=0.18, lw=2.5)
        ax.text(x_min + (x_max - x_min) * 0.13, y_mid, z_floor + 0.012,
                "+X  forward", color="#d62728", fontsize=9, weight="bold")

    tmp_dir = out_path.parent / f"_frames_{out_path.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    stride = max(1, T // 100)
    frame_paths = []

    for t_idx in range(0, T, stride):
        fig = plt.figure(figsize=(8.5, 4.5), dpi=100)
        ax = fig.add_subplot(projection="3d")
        ax.set_proj_type("ortho")
        _draw_floor_and_arrow(ax)
        # Body particles (opaque)
        ax.scatter(px[t_idx], py[t_idx], pz[t_idx],
                   s=6.0, c=colors, alpha=1.0, edgecolors="none", zorder=2)
        # COM trail
        if t_idx > 0:
            ax.plot(com_x[: t_idx + 1], com_y[: t_idx + 1], com_z[: t_idx + 1],
                    color="red", lw=1.8, alpha=0.9, zorder=3)
            # Shadow on the floor
            ax.plot(com_x[: t_idx + 1], com_y[: t_idx + 1],
                    np.full(t_idx + 1, z_floor),
                    color="#ff7f0e", lw=1.2, ls="--", alpha=0.8, zorder=1)
        ax.scatter([com_x[t_idx]], [com_y[t_idx]], [com_z[t_idx]],
                   s=42, c="red", marker="o", edgecolors="white", lw=1.0, zorder=4)

        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_floor - 0.01, z_top)
        try:
            ax.set_box_aspect((x_max - x_min, y_max - y_min, z_top - z_floor))
        except Exception:
            pass
        ax.set_xlabel("X  (forward)", fontsize=9, labelpad=2)
        ax.set_ylabel("Z  (side)",    fontsize=9, labelpad=2)
        ax.set_zlabel("Y  (up)",      fontsize=9, labelpad=2)
        ax.tick_params(axis="both", labelsize=7, pad=-1)
        ax.view_init(elev=22, azim=-60)
        forward = com_x[t_idx] - com_x[0]
        ax.set_title(
            f"{title}\nstep={t_idx}/{T}   return={return_value:.3f}   "
            f"forward=+{forward:.3f}",
            fontsize=10,
        )
        fpath = tmp_dir / f"f_{t_idx:04d}.png"
        fig.savefig(fpath, bbox_inches="tight"); plt.close(fig)
        frame_paths.append(fpath)

    frames = [iio.imread(p) for p in frame_paths]
    iio.imwrite(out_path, frames, duration=60, loop=0)
    for p in frame_paths: p.unlink(missing_ok=True)
    try: tmp_dir.rmdir()
    except OSError: pass

    # Static summary
    summary_path = out_path.with_suffix(".summary.png")
    fig = plt.figure(figsize=(13, 4.5), dpi=110)
    for i, (t_idx, lab) in enumerate(zip([0, T - 1], ["start", "end"])):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.set_proj_type("ortho")
        _draw_floor_and_arrow(ax)
        ax.scatter(px[t_idx], py[t_idx], pz[t_idx],
                   s=6, c=colors, alpha=1.0, edgecolors="none")
        ax.plot(com_x[: t_idx + 1], com_y[: t_idx + 1], com_z[: t_idx + 1],
                color="red", lw=1.8, alpha=0.9)
        ax.plot(com_x[: t_idx + 1], com_y[: t_idx + 1],
                np.full(t_idx + 1, z_floor),
                color="#ff7f0e", lw=1.2, ls="--")
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_floor - 0.01, z_top)
        try:
            ax.set_box_aspect((x_max - x_min, y_max - y_min, z_top - z_floor))
        except Exception:
            pass
        ax.tick_params(axis="both", labelsize=7)
        ax.set_xlabel("X  (forward)", fontsize=9)
        ax.set_ylabel("Z  (side)",    fontsize=9)
        ax.set_zlabel("Y  (up)",      fontsize=9)
        ax.view_init(elev=22, azim=-60)
        ax.set_title(f"{lab}  step={t_idx}", fontsize=11, weight="bold")
    fig.suptitle(
        f"{title}   return={return_value:.3f}   "
        f"forward=+{com_x[-1]-com_x[0]:.3f}",
        fontsize=12, weight="bold",
    )
    fig.tight_layout(); fig.savefig(summary_path, bbox_inches="tight"); plt.close(fig)


def _capture_rollout_worker(
    task_id: str,
    x_star: np.ndarray,
    phi_star: np.ndarray,
    runtime_config_kwargs: dict,
    project_root_str: str,
    out_queue,
):
    """Subprocess body: runs one rollout and pushes (positions, return) to queue.

    Lives in its own process so its Taichi CUDA context cannot collide with
    the parent's JAX CUDA context (which deadlocks at the driver futex level).
    """
    import os
    os.environ.setdefault("SOFTZOO_SKIP_OCCUPANCY_KERNELS", "1")
    import numpy as _np
    from genedynamics.envs.external.softzoo.bootstrap import ensure_softzoo_on_path
    from genedynamics.envs.external.softzoo.adapters import (
        make_softzoo_env, encode_morphology, encode_controller,
    )
    from genedynamics.envs.external.softzoo.task_registry import get_task_spec
    from genedynamics.envs.external.softzoo.config import SoftZooRuntimeConfig
    ensure_softzoo_on_path(project_root_str)
    rc = SoftZooRuntimeConfig(
        project_root=Path(project_root_str),
        ti_arch=runtime_config_kwargs.get("ti_arch"),
        ti_device_memory_fraction=runtime_config_kwargs.get("ti_device_memory_fraction"),
    )
    spec = get_task_spec(task_id)
    env = make_softzoo_env(
        task_spec=spec, fidelity_spec=None, mode_spec=None, runtime_config=rc,
    )
    ctrl = encode_controller(phi_star, spec, env)
    dsg = encode_morphology(x_star, spec, env=env)
    obs = env.reset(dsg); ctrl.reset()
    positions = []
    ret = 0.0
    for _ in range(spec.max_steps):
        s = env.sim.solver.current_s
        rx = env.design_space.get_x(s)
        rx_np = rx.numpy() if hasattr(rx, "numpy") else _np.asarray(rx)
        positions.append(rx_np.copy())
        act = ctrl(s, obs)
        obs, r, done, info = env.step(act)
        ret += float(r)
        if done:
            break
    env.close()
    out_queue.put((_np.stack(positions, axis=0).astype(_np.float32), float(ret)))


def _capture_rollout_jax_mpm(
    x_star: np.ndarray,
    phi_star: np.ndarray,
    evaluator_params: dict,
    replay_friction: float | None = None,
) -> tuple[np.ndarray, float]:
    """Capture rollout for jax_mpm backend. No subprocess needed — JAX MPM is pure
    JAX so it shares the main process's GPU context freely."""
    import jax.numpy as jnp
    from genedynamics.envs.external.jax_mpm.scene import (
        MPMConfig, build_scene, rollout_with_positions,
    )

    mpm_kwargs = {
        "n_grid": int(evaluator_params.get("n_grid", 64)),
        "shaping_weight": float(evaluator_params.get("reward_shaping_weight", 100.0)),
    }
    # Forward ALL MPMConfig-compatible fields so the GIF replay uses the same
    # physics the planner saw (act_strength_base, scale, voxel_dims, etc.).
    for k in ("voxel_dims", "act_strength_base", "scale", "dt", "gravity",
              "p_vol", "friction_coeff", "actuation_strength_scale",
              "backward_penalty_weight"):
        v = evaluator_params.get(k)
        if v is not None:
            if k == "voxel_dims":
                mpm_kwargs[k] = tuple(int(d) for d in v)
            else:
                mpm_kwargs[k] = float(v)
    cfg = MPMConfig(**{k: v for k, v in mpm_kwargs.items()
                       if k in MPMConfig.__dataclass_fields__})
    scene = build_scene(cfg)
    # Use the friction that MBD planned against if caller provides it
    # (important when num_modes=1: the planner only saw _mode_friction[0]=0.3,
    # so replaying at fr=0.45 silently breaks the gait).
    fr = 0.45 if replay_friction is None else float(replay_friction)
    num_env_steps = 200  # match softzoo default for gif fidelity

    xm = jnp.asarray(x_star, dtype=jnp.float32)
    ph = jnp.asarray(phi_star, dtype=jnp.float32)
    fr_j = jnp.asarray(fr, dtype=jnp.float32)
    reward, disp, com_traj, particle_traj, mass_field = rollout_with_positions(
        xm, ph, fr_j, scene, cfg, num_env_steps,
    )
    reward.block_until_ready()
    # Aggregate particles → per-voxel centroids so the GIF matches the morphology
    # voxel grid (48 deformable voxels) rather than thousands of MPM particles.
    particle_traj_np = np.asarray(particle_traj, dtype=np.float32)  # (T, N, 3)
    voxel_id = np.asarray(scene.voxel_id)                           # (N,)
    mass_np = np.asarray(mass_field, dtype=np.float32)              # (N,)
    n_voxels = int(scene.n_voxels)

    # Per-voxel occupancy (mean of per-particle mass for particles in that voxel).
    occ_per_voxel = np.zeros(n_voxels, dtype=np.float32)
    count_per_voxel = np.zeros(n_voxels, dtype=np.int32)
    np.add.at(occ_per_voxel, voxel_id, mass_np)
    np.add.at(count_per_voxel, voxel_id, 1)
    occ_per_voxel = occ_per_voxel / np.maximum(count_per_voxel, 1)

    # Per-voxel centroid per timestep: mass-weighted mean of particles in that voxel.
    T = particle_traj_np.shape[0]
    vox_centroids = np.zeros((T, n_voxels, 3), dtype=np.float32)
    # Weights for each particle = mass (1-hot-summed into its voxel).
    w = np.maximum(mass_np, 1e-6)  # avoid div-by-zero on empty voxels
    for t in range(T):
        weighted_pos = particle_traj_np[t] * w[:, None]            # (N, 3)
        num = np.zeros((n_voxels, 3), dtype=np.float32)
        den = np.zeros(n_voxels, dtype=np.float32)
        np.add.at(num, voxel_id, weighted_pos)
        np.add.at(den, voxel_id, w)
        vox_centroids[t] = num / np.maximum(den[:, None], 1e-8)
    voxel_dims = np.asarray(cfg.voxel_dims, dtype=np.int32)

    return {
        "voxel_centroids": vox_centroids,   # (T, n_voxels, 3)
        "voxel_occupancy": occ_per_voxel,   # (n_voxels,)
        "voxel_dims": voxel_dims,           # (vx, vy, vz)
        "reward": float(reward),
    }


def _capture_rollout(
    task_id: str,
    x_star: np.ndarray,
    phi_star: np.ndarray,
    runtime_config_kwargs: dict,
    project_root: Path,
) -> tuple[np.ndarray, float]:
    """Run capture in a spawn-mode subprocess to isolate Taichi from parent's JAX."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(
        target=_capture_rollout_worker,
        args=(task_id, x_star, phi_star, runtime_config_kwargs, str(project_root), q),
        daemon=False,
    )
    proc.start()
    try:
        positions, ret = q.get(timeout=300)
    finally:
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
    return positions, ret


def main() -> int:
    parser = argparse.ArgumentParser(description="Run co-design + render GIF")
    parser.add_argument("config", type=str)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-train", action="store_true",
                        help="Reuse existing results.json (only render GIF)")
    parser.add_argument("--gif-name", type=str, default="crawling_best.gif")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (root / config_path).resolve()
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    if args.seed is not None:
        data["seeds"] = [args.seed]

    # Pull runtime config (ti_arch, mem fraction, etc.)
    evaluator_params = data.get("evaluator_runtime", data.get("evaluator_params", {})) or {}
    if data.get("softzoo_ti_arch"):
        evaluator_params = dict(evaluator_params, ti_arch=data["softzoo_ti_arch"])
    if data.get("ti_device_memory_fraction") is not None:
        evaluator_params = dict(
            evaluator_params, ti_device_memory_fraction=float(data["ti_device_memory_fraction"])
        )

    output_dir = Path(data.get("output_dir", "results/co_design"))
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    if not args.skip_train:
        from genedynamics.experiments.framework.baseline_platform import (
            BaselineExperimentPlatform,
            BaselineExperimentConfig,
        )
        import genedynamics.experiments.framework.baselines  # noqa: F401

        cfg = BaselineExperimentConfig(
            baseline_name=data.get("baseline_name", "mrmfmbd"),
            task_domain=data.get("task_domain", "softzoo"),
            task_id=data.get("task_id", "crawling_ground"),
            seeds=data.get("seeds", [0]),
            output_dir=str(output_dir),
            cache_dir=data.get("cache_dir", data.get("checkpoint_dir")),
            scheduler_config=data.get("scheduler_config"),
            method_params=data.get("method_params", {}),
            baseline_params=data.get("baseline_params", {}),
            evaluator_params=evaluator_params,
            save_gif=data.get("save_gif", True),
        )
        platform = BaselineExperimentPlatform(cfg, project_root=root)
        # MBD and its siblings are zero-shot samplers, not learners — no
        # "training" happens here. Frame the wall time as planning time.
        verb = "Planning" if cfg.baseline_name in ("mrmfmbd", "mbd") else "Running"
        print(f"{verb}: baseline={cfg.baseline_name} task={cfg.task_id} seeds={cfg.seeds}")
        t0 = time.time()
        results = platform.run_all()
        print(f"{verb} complete in {time.time()-t0:.1f}s")
    else:
        import json
        with open(output_dir / "results.json") as f:
            results = json.load(f)

    # Pick seed with the best final return
    def _get_return(r):
        res = r.get("result", {})
        return float(res.get("return_", 0.0))
    best = max(results, key=_get_return)
    res = best.get("result", {})
    x_star = np.asarray(res["x"], dtype=np.float32)
    phi_star = np.asarray(res["phi"], dtype=np.float32)
    print(f"Best seed={best['seed']} return={_get_return(best):.4f} x={x_star} phi_len={len(phi_star)}")

    print("Replaying best theta with trajectory capture...")
    t0 = time.time()
    task_domain = data.get("task_domain", "softzoo")
    if task_domain == "jax_mpm":
        num_modes = int((data.get("method_params") or {}).get("num_modes", 4))
        _mode_friction_table = [float(f) for f in evaluator_params.get(
            "mode_friction", [0.3, 0.4, 0.5, 0.6]
        )]
        replay_fr = _mode_friction_table[0] if num_modes == 1 else None
        capture = _capture_rollout_jax_mpm(
            x_star=x_star, phi_star=phi_star,
            evaluator_params=evaluator_params,
            replay_friction=replay_fr,
        )
        vox_centroids = capture["voxel_centroids"]
        vox_occ = capture["voxel_occupancy"]
        vox_dims = capture["voxel_dims"]
        replay_return = capture["reward"]
        T = vox_centroids.shape[0]
        print(f"Captured {T} frames in {time.time()-t0:.1f}s, replay_return={replay_return:.4f}")
        gif_path = output_dir / args.gif_name
        _render_voxel_gif(
            vox_centroids=vox_centroids,
            vox_occupancy=vox_occ,
            voxel_dims=tuple(int(d) for d in vox_dims),
            out_path=gif_path,
            return_value=replay_return,
            title=f"{data.get('name', config_path.stem)} seed={best['seed']}",
        )
    else:
        positions, replay_return = _capture_rollout(
            task_id=data.get("task_id", "crawling_ground"),
            x_star=x_star, phi_star=phi_star,
            runtime_config_kwargs=evaluator_params,
            project_root=root,
        )
        print(f"Captured {positions.shape[0]} frames in {time.time()-t0:.1f}s, replay_return={replay_return:.4f}")
        gif_path = output_dir / args.gif_name
        _render_gif(
            positions_over_time=positions,
            out_path=gif_path,
            return_value=replay_return,
            title=f"{data.get('name', config_path.stem)} seed={best['seed']}",
        )
    print(f"GIF saved: {gif_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
