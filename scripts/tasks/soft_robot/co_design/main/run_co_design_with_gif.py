#!/usr/bin/env python3
"""
Run co-design experiment from YAML config, then render best-theta rollout as GIF.

Wraps run_co_design.py; after planning, rebuilds the jax_mpm scene, rolls out
best (x*, phi*) while capturing per-voxel centroids, and renders a 3D
matplotlib GIF.
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

    cx_all = vox_centroids[..., 0]
    cy_all = vox_centroids[..., 2]
    cz_all = vox_centroids[..., 1]

    com = vox_centroids.mean(axis=1)
    com_px = com[:, 0]; com_py = com[:, 2]; com_pz = com[:, 1]

    x_min = float(cx_all.min() - 0.03); x_max = float(cx_all.max() + 0.03)
    y_min = float(cy_all.min() - 0.02); y_max = float(cy_all.max() + 0.02)
    z_floor = float(cz_all.min() - 0.012)
    z_top   = float(cz_all.max() + 0.03)

    cmap = plt.get_cmap("viridis")
    pitch_x = (cx_all.max() - cx_all.min()) / max(vx - 1, 1)
    pitch_y = (cy_all.max() - cy_all.min()) / max(vz - 1, 1)
    pitch_z = (cz_all.max() - cz_all.min()) / max(vy - 1, 1)
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
        for v in range(n_voxels):
            occ = float(vox_occupancy[v])
            if occ < 0.15:
                continue
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


def _capture_rollout_jax_mpm(
    x_star: np.ndarray,
    phi_star: np.ndarray,
    evaluator_params: dict,
    replay_friction: float | None = None,
) -> dict:
    """Capture rollout for jax_mpm backend. Pure JAX so no subprocess needed."""
    import jax.numpy as jnp
    from genedynamics.envs.external.jax_mpm.scene import (
        MPMConfig, build_scene, rollout_with_positions,
    )

    mpm_kwargs = {
        "n_grid": int(evaluator_params.get("n_grid", 64)),
        "shaping_weight": float(evaluator_params.get("reward_shaping_weight", 100.0)),
    }
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
    # Use the friction the planner saw when num_modes=1, otherwise default to 0.45
    # (matches MPMConfig default and the planner's mid-range mode_friction).
    fr = 0.45 if replay_friction is None else float(replay_friction)
    num_env_steps = 200

    xm = jnp.asarray(x_star, dtype=jnp.float32)
    ph = jnp.asarray(phi_star, dtype=jnp.float32)
    fr_j = jnp.asarray(fr, dtype=jnp.float32)
    reward, disp, com_traj, particle_traj, mass_field = rollout_with_positions(
        xm, ph, fr_j, scene, cfg, num_env_steps,
    )
    reward.block_until_ready()
    particle_traj_np = np.asarray(particle_traj, dtype=np.float32)
    voxel_id = np.asarray(scene.voxel_id)
    mass_np = np.asarray(mass_field, dtype=np.float32)
    n_voxels = int(scene.n_voxels)

    occ_per_voxel = np.zeros(n_voxels, dtype=np.float32)
    count_per_voxel = np.zeros(n_voxels, dtype=np.int32)
    np.add.at(occ_per_voxel, voxel_id, mass_np)
    np.add.at(count_per_voxel, voxel_id, 1)
    occ_per_voxel = occ_per_voxel / np.maximum(count_per_voxel, 1)

    T = particle_traj_np.shape[0]
    vox_centroids = np.zeros((T, n_voxels, 3), dtype=np.float32)
    w = np.maximum(mass_np, 1e-6)
    for t in range(T):
        weighted_pos = particle_traj_np[t] * w[:, None]
        num = np.zeros((n_voxels, 3), dtype=np.float32)
        den = np.zeros(n_voxels, dtype=np.float32)
        np.add.at(num, voxel_id, weighted_pos)
        np.add.at(den, voxel_id, w)
        vox_centroids[t] = num / np.maximum(den[:, None], 1e-8)
    voxel_dims = np.asarray(cfg.voxel_dims, dtype=np.int32)

    return {
        "voxel_centroids": vox_centroids,
        "voxel_occupancy": occ_per_voxel,
        "voxel_dims": voxel_dims,
        "reward": float(reward),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run co-design + render GIF (jax_mpm)")
    parser.add_argument("config", type=str)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-train", action="store_true",
                        help="Reuse existing results.json (only render GIF)")
    parser.add_argument("--gif-name", type=str, default="crawling_best.gif")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[5]
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

    evaluator_params = data.get("evaluator_runtime", data.get("evaluator_params", {})) or {}

    task_domain = data.get("task_domain", "jax_mpm")
    if task_domain != "jax_mpm":
        raise ValueError(
            f"This GIF runner only supports task_domain=jax_mpm, got {task_domain!r}."
        )

    output_dir = Path(data.get("output_dir", "results/soft_robot/co_design"))
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    if not args.skip_train:
        from genedynamics.experiments.framework.baseline_platform import (
            BaselineExperimentPlatform,
            BaselineExperimentConfig,
        )
        import genedynamics.solvers.single.codesign_solvers  # noqa: F401 — register co-design solvers

        cfg = BaselineExperimentConfig(
            baseline_name=data.get("baseline_name", "mrmfmbd"),
            task_domain=task_domain,
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
        verb = "Planning" if cfg.baseline_name in ("mrmfmbd", "mbd") else "Running"
        print(f"{verb}: baseline={cfg.baseline_name} task={cfg.task_id} seeds={cfg.seeds}")
        t0 = time.time()
        results = platform.run_all()
        print(f"{verb} complete in {time.time()-t0:.1f}s")
    else:
        import json
        with open(output_dir / "results.json") as f:
            results = json.load(f)

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
    print(f"GIF saved: {gif_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
