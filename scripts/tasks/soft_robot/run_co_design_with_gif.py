#!/usr/bin/env python3
"""
Run co-design experiment from YAML config, then render best-theta rollout as GIF.

Wraps run_co_design.py; after training, rebuilds env, rolls out best (x*, phi*)
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


def _render_gif(positions_over_time: np.ndarray, out_path: Path, return_value: float, title: str) -> None:
    """
    positions_over_time: (T, N, 3) particle positions.
    Side-view scatter animation. Particles colored by initial x to make
    deformation visible; camera tracks COM so motion stays centered;
    COM trajectory drawn as a red trail.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v3 as iio

    pos = positions_over_time
    T, N, _ = pos.shape

    init_x = pos[0, :, 0]
    rmin, rmax = float(init_x.min()), float(init_x.max())
    norm_x = (init_x - rmin) / max(rmax - rmin, 1e-6)
    cmap = plt.get_cmap("turbo")
    colors = cmap(norm_x)

    com = pos.mean(axis=1)  # (T, 3)
    body_extent_x = rmax - rmin
    body_extent_y = float(pos[0, :, 1].max() - pos[0, :, 1].min())
    half_w = max(body_extent_x * 1.5, 0.08)
    half_h = max(body_extent_y * 1.5, 0.06)

    y_floor = float(pos[..., 1].min())
    y_top = float(com[:, 1].max() + body_extent_y * 1.5)

    tmp_dir = out_path.parent / f"_frames_{out_path.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    stride = max(1, T // 100)
    frame_paths = []
    for t_idx in range(0, T, stride):
        fig, ax = plt.subplots(figsize=(7, 3.0), dpi=100)
        ax.axhspan(y_floor - 0.02, y_floor, color="0.85", zorder=0)  # ground band
        if t_idx > 0:
            ax.plot(com[: t_idx + 1, 0], com[: t_idx + 1, 1], color="red", lw=1.5, alpha=0.7, zorder=1)
        ax.scatter(pos[t_idx, :, 0], pos[t_idx, :, 1], s=4.5, c=colors, alpha=0.9, edgecolors="none", zorder=2)
        ax.scatter([com[t_idx, 0]], [com[t_idx, 1]], s=40, c="red", marker="o", edgecolors="white", lw=1.0, zorder=3)
        cx = com[t_idx, 0]
        ax.set_xlim(cx - half_w, cx + half_w)
        ax.set_ylim(y_floor - 0.01, y_top)
        ax.set_aspect("equal")
        ax.set_xlabel("x (forward)")
        ax.set_ylabel("y (up)")
        forward = com[t_idx, 0] - com[0, 0]
        ax.set_title(
            f"{title}\nstep={t_idx}/{T}  return={return_value:.3f}  forward={forward:+.3f}"
        )
        fig.tight_layout()
        fpath = tmp_dir / f"f_{t_idx:04d}.png"
        fig.savefig(fpath)
        plt.close(fig)
        frame_paths.append(fpath)

    frames = [iio.imread(p) for p in frame_paths]
    iio.imwrite(out_path, frames, duration=60, loop=0)
    for p in frame_paths:
        p.unlink(missing_ok=True)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    # Also dump a static summary plot (start vs end side-by-side + COM trajectory)
    summary_path = out_path.with_suffix(".summary.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), dpi=110)
    for ax, t_idx, label in zip(axes, [0, T - 1], ["start", "end"]):
        ax.axhspan(y_floor - 0.02, y_floor, color="0.85")
        ax.scatter(pos[t_idx, :, 0], pos[t_idx, :, 1], s=5, c=colors, alpha=0.9, edgecolors="none")
        ax.plot(com[: t_idx + 1, 0], com[: t_idx + 1, 1], color="red", lw=1.5, alpha=0.7)
        ax.set_aspect("equal")
        ax.set_xlim(min(com[:, 0].min(), pos[0, :, 0].min()) - 0.05,
                    max(com[:, 0].max(), pos[-1, :, 0].max()) + 0.05)
        ax.set_ylim(y_floor - 0.01, y_top)
        ax.set_title(f"{label}  step={t_idx}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(f"{title}  return={return_value:.3f}  forward={com[-1,0]-com[0,0]:+.3f}")
    fig.tight_layout()
    fig.savefig(summary_path)
    plt.close(fig)


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
    cfg = MPMConfig(**{k: v for k, v in mpm_kwargs.items()
                       if k in MPMConfig.__dataclass_fields__})
    scene = build_scene(cfg)
    fr = 0.45  # mid friction for replay — sweet spot for stick-slip peristalsis
    num_env_steps = 200  # match softzoo default for gif fidelity

    xm = jnp.asarray(x_star, dtype=jnp.float32)
    ph = jnp.asarray(phi_star, dtype=jnp.float32)
    fr_j = jnp.asarray(fr, dtype=jnp.float32)
    reward, disp, com_traj, particle_traj, mass_field = rollout_with_positions(
        xm, ph, fr_j, scene, cfg, num_env_steps,
    )
    reward.block_until_ready()
    # Filter ghost particles (mass < 0.5 from voxel occupancy < 0.5) so the
    # gif shows only the materialized robot.
    mask = np.asarray(mass_field) > 0.5
    if mask.sum() == 0:  # degenerate empty robot — show all
        mask = np.ones_like(mask, dtype=bool)
    particles_visible = np.asarray(particle_traj, dtype=np.float32)[:, mask, :]
    return particles_visible, float(reward)


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
            checkpoint_dir=data.get("checkpoint_dir"),
            scheduler_config=data.get("scheduler_config"),
            method_params=data.get("method_params", {}),
            baseline_params=data.get("baseline_params", {}),
            evaluator_params=evaluator_params,
            save_gif=data.get("save_gif", True),
        )
        platform = BaselineExperimentPlatform(cfg, project_root=root)
        print(f"Training: baseline={cfg.baseline_name} task={cfg.task_id} seeds={cfg.seeds}")
        t0 = time.time()
        results = platform.run_all()
        print(f"Training complete in {time.time()-t0:.1f}s")
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
        positions, replay_return = _capture_rollout_jax_mpm(
            x_star=x_star, phi_star=phi_star,
            evaluator_params=evaluator_params,
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
