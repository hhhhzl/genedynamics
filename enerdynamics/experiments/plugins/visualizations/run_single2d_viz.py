"""
Single script to generate all single2d result visualizations.

Run from project root (with the same env that has enerdynamics deps, e.g. jax), e.g.:
  python -m enerdynamics.experiments.plugins.visualizations.run_single2d_viz --which all
  python -m enerdynamics.experiments.plugins.visualizations.run_single2d_viz --which cost

Outputs go to results/single2d/visualizations/ (cost_vs_step/, best_trajectories/, cfsmbd_adaptive/, ssr_heatmap.png, time_per_level/).
Use --config CONFIG so best_trajectories obstacles match results/env_preview (same YAML as scripts/visualize_env_levels_seeds.py).
Uncertainty band: SEM × UNCERTAINTY_MULTIPLIER (default 2.0). Transparency: STD_BAND_ALPHA. Algorithm colors: ALGO_COLORS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from ...common.visualization import draw_obstacles
from ...common.obstacle_generation import generate_box2d_obstacles

# ---------------------------------------------------------------------------
# Paths and constants (adjust as needed)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_ROOT = _PROJECT_ROOT / "results" / "single2d"
OUT_DIR = RESULTS_ROOT / "visualizations"

ALGOS_DISPLAY_NAMES = {
    "mbd": "MBD",
    "ebmbd": "EB-MBD",
    "mdoc": "MDOC",
    "mdcoas-a": "MD-COAS-A",
    "mdcoas-f": "MD-COAS-F",
    "mdcoas": "MD-COAS",
}

ALGOS_ORDER = [
    "mbd",
    "ebmbd",
    "mdoc",
    "mdcoas-a",
    "mdcoas-f",
    "mdcoas",
]
CFSMBD_ALGOS = ["mdcoas-a", "mdcoas-f", "mdcoas"]
LEVELS = list(range(1, 11))  # 1..10
SEEDS = list(range(10))  # 0..9

# Colors for each algorithm (same order as ALGOS_ORDER). Easy to change.
ALGO_COLORS = [
    "#CC79A7",  # mbd
    "#9467bd",  # ebmbd
    "#2ca02c",  # mdoc
    "#d62728",  # mdcoas-a
    "#ff7f0e",  # mdcoas-f
    "#1f77b4",  # mdcoas
]

# Uncertainty band transparency (lower = lighter). Used in cost vs step, adaptive, etc.
STD_BAND_ALPHA = 0.12
# Multiplier for SEM band width (2.0 = ±2×SEM, wider bands)
UNCERTAINTY_MULTIPLIER = 2.0

# Box2D obstacle config (fallback when --config not used; match single2d YAML)
BOX2D_CONFIG = {
    "robot_radius": 0.05,
    "obstacle_radius_scale": 1.0,
    "min_obstacle_margin": 2.4 * 0.05,
    "p_max": 2.0,
    "map_bounds": {
        "x_min": -1.5,
        "x_max": 1.0,
        "y_min": -2.0,
        "y_max": 0.5,
    },
    "enable_connectivity_check": True,
    "enable_nonconvexity_check": True,
}

# LaTeX labels for adaptive metrics (matplotlib math mode)
ADAPTIVE_METRIC_LATEX = {
    "r_k": r"$r_k$",
    "v_rate": r"$v_{\mathrm{rate}}$",
    "v_mean": r"$v_{\mathrm{mean}}$",
    "c_k": r"$c_k$",
    "nu": r"$\nu$",
    "p_k": r"$p_k$",
    "rho": r"$\rho$",
    "topK": r"$\mathrm{topK}$",
    "I_QP": r"$I_{\mathrm{QP}}$",
    "eps": r"$\varepsilon$",
    "lambda": r"$\lambda$",
}


def _metric_label(name: str) -> str:
    """LaTeX label for adaptive metric name."""
    return ADAPTIVE_METRIC_LATEX.get(name, name)


def _adaptive_metrics_raw_to_array(raw: list) -> np.ndarray:
    """Convert metrics from JSON (possibly with null for c_k/nu) to float64 array; null → nan."""
    if not raw:
        return np.array([], dtype=np.float64)

    def replace_none(x: Any) -> Any:
        if isinstance(x, (list, tuple)):
            return [replace_none(v) for v in x]
        if x is None:
            return float("nan")
        return x

    return np.asarray(replace_none(raw), dtype=np.float64)


def _algo_color(algo: str) -> str:
    if algo in ALGOS_ORDER:
        return ALGO_COLORS[ALGOS_ORDER.index(algo)]
    return "#333333"


def _algo_display_name(algo: str) -> str:
    return ALGOS_DISPLAY_NAMES.get(algo, algo)


def _set_diffusion_axis(ax: plt.Axes, n_steps: int) -> None:
    # Position 0 = noisiest (step 100), position n_steps-1 = cleanest (step 1)
    # Always include 100 at 0 and 1 at n_steps-1; add 80,60,40,20 where they fit
    tick_positions = [0]
    tick_labels = ["100"]
    for k in [80, 60, 40, 20]:
        pos = n_steps - k
        if 0 < pos < n_steps - 1:
            tick_positions.append(pos)
            tick_labels.append(str(k))
    tick_positions.append(n_steps - 1)
    tick_labels.append("1")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Diffusion Denoising Step", fontsize=20)
    ax.tick_params(axis="x", labelsize=20)
    # Explicit xlim padding so edge ticks (100, 1) are fully visible
    pad = max(3, int(n_steps * 0.03))
    ax.set_xlim(-pad, n_steps - 1 + pad)


# ---------------------------------------------------------------------------
# Plot 1: Cost vs diffusion step (per level, all algos; approach A)
# ---------------------------------------------------------------------------
def plot_cost_vs_step(results_root: Path, out_dir: Path) -> None:
    """For each level, plot each algorithm's mean cost vs diffusion step with SEM band (approach A)."""
    subdir = out_dir / "cost_vs_step"
    subdir.mkdir(parents=True, exist_ok=True)

    for level in LEVELS:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        n_steps = None

        for algo in ALGOS_ORDER:
            cost_dir = results_root / algo / f"level_{level}"
            if not cost_dir.exists():
                continue

            # Per-seed mean over modes, then mean and SEM over seeds (approach A)
            seed_means_list = []  # list of (n_steps,) arrays

            for seed in SEEDS:
                cost_path = cost_dir / f"seed_{seed}" / "cost" / "cost.json"
                if not cost_path.exists():
                    continue
                with open(cost_path) as f:
                    data = json.load(f)
                cost_per_mode = data.get("cost_per_mode", [])
                if not cost_per_mode:
                    continue
                arr = np.array(cost_per_mode, dtype=np.float64)  # (M, H)
                seed_mean = np.mean(arr, axis=0)  # (H,)
                seed_means_list.append(seed_mean)
                if n_steps is None:
                    n_steps = arr.shape[1]

            if not seed_means_list:
                continue
            seed_means = np.array(seed_means_list)  # (10, H)
            mean_curve = np.mean(seed_means, axis=0)
            n_seeds = seed_means.shape[0]
            sem_curve = np.std(seed_means, axis=0) / np.sqrt(n_seeds) * UNCERTAINTY_MULTIPLIER
            x = np.arange(len(mean_curve))
            color = _algo_color(algo)
            ax.fill_between(x, mean_curve - sem_curve, mean_curve + sem_curve, alpha=STD_BAND_ALPHA, color=color)
            ax.plot(x, mean_curve, color=color, linewidth=2.0, label=_algo_display_name(algo))

        if n_steps is None:
            n_steps = 100
        _set_diffusion_axis(ax, n_steps)
        ax.set_ylabel("Cost")
        ax.set_title(f"Cost vs Diffusion Step (Level {level})")
        ax.legend(loc="best", fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        fig.savefig(subdir / f"level_{level}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  cost_vs_step: level_{level}.png")

    # One summary figure: average over all levels (and seeds)
    algo_curves_list: dict[str, list[np.ndarray]] = {}
    for algo in ALGOS_ORDER:
        all_curves = []
        for level in LEVELS:
            cost_dir = results_root / algo / f"level_{level}"
            if not cost_dir.exists():
                continue
            for seed in SEEDS:
                cost_path = cost_dir / f"seed_{seed}" / "cost" / "cost.json"
                if not cost_path.exists():
                    continue
                with open(cost_path) as f:
                    data = json.load(f)
                cost_per_mode = data.get("cost_per_mode", [])
                if not cost_per_mode:
                    continue
                arr = np.array(cost_per_mode, dtype=np.float64)
                seed_mean = np.mean(arr, axis=0)
                all_curves.append(seed_mean)
        if all_curves:
            algo_curves_list[algo] = all_curves
    n_steps_ref = 100
    if algo_curves_list:
        n_steps_ref = min(c.shape[0] for curves in algo_curves_list.values() for c in curves)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for algo, all_curves in algo_curves_list.items():
        stack = np.array([c[:n_steps_ref] for c in all_curves])
        mean_curve = np.mean(stack, axis=0)
        std_curve = np.std(stack, axis=0)
        x = np.arange(len(mean_curve))
        color = _algo_color(algo)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=STD_BAND_ALPHA, color=color)
        ax.plot(x, mean_curve, color=color, linewidth=3.0, label=_algo_display_name(algo), marker="o", markersize=1)
    _set_diffusion_axis(ax, n_steps_ref)
    ax.set_ylabel("Cost", fontsize=20)
    ax.set_title("Mean Cost vs Diffusion Step for All Levels & Seeds", fontsize=21, fontweight="bold")
    ax.legend(loc="best", fontsize=15)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="y", labelsize=20)
    #ax.grid(True, alpha=0.3)
    fig.savefig(subdir / "all_levels_mean.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  cost_vs_step: all_levels_mean.png")

    # Mean cost vs level: x=level 1..10, y=mean cost (final step) + SEM band
    x_levels = np.array(LEVELS, dtype=np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for algo in ALGOS_ORDER:
        means = []
        sems = []
        for level in LEVELS:
            cost_dir = results_root / algo / f"level_{level}"
            seed_vals = []
            if cost_dir.exists():
                for seed in SEEDS:
                    cost_path = cost_dir / f"seed_{seed}" / "cost" / "cost.json"
                    if cost_path.exists():
                        with open(cost_path) as f:
                            data = json.load(f)
                        cost_per_mode = data.get("cost_per_mode", [])
                        if cost_per_mode:
                            arr = np.array(cost_per_mode, dtype=np.float64)
                            final_costs = arr[:, -1]
                            seed_vals.append(np.mean(final_costs))
            if seed_vals:
                n_s = len(seed_vals)
                means.append(np.mean(seed_vals))
                sems.append((np.std(seed_vals) / np.sqrt(n_s) if n_s > 0 else 0.0) * UNCERTAINTY_MULTIPLIER)
            else:
                means.append(np.nan)
                sems.append(np.nan)
        means = np.array(means)
        sems = np.array(sems)
        color = _algo_color(algo)
        ax.fill_between(x_levels, means - sems, means + sems, alpha=STD_BAND_ALPHA, color=color)
        ax.plot(x_levels, means, color=color, linewidth=2.0, label=_algo_display_name(algo), marker="o", markersize=10)
    ax.set_xticks(LEVELS)
    ax.set_xticklabels([f"L{l}" for l in LEVELS], rotation=45, ha="right", rotation_mode="anchor", fontsize=20)
    ax.axvline(x=3.5, color="black", linestyle="--", linewidth=3, alpha=0.7)
    ax.axvline(x=6.5, color="black", linestyle="--", linewidth=3, alpha=0.7)
    ax.annotate("Easy Maps", xy=(1.0 / len(LEVELS), -0.18), xycoords="axes fraction", ha="center", fontsize=20, color="black")
    ax.annotate("Constrained Maps", xy=(4.5 / len(LEVELS), -0.18), xycoords="axes fraction", ha="center", fontsize=20, color="black")
    ax.annotate("Union Maps", xy=(8.5 / len(LEVELS), -0.18), xycoords="axes fraction", ha="center", fontsize=20, color="black")
    # ax.set_xlabel("Level")
    ax.set_ylabel("Cost", fontsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_title("Final Diffusion Step Mean Cost Over 10 Seeds", fontsize=22, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", fontsize=15)
    # ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(subdir / "mean_cost_vs_level.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  cost_vs_step: mean_cost_vs_level.png")


def _get_env_setup_from_config(config_path: Path, level: int, seed: int):
    """
    Get start_2d, goal_2d, obstacles for (level, seed) using same logic as env_preview.
    Returns (start_2d, goal_2d, obstacles) or (None, None, None) on failure.
    """
    try:
        from ...framework import ExperimentRunner, ExperimentConfig
        from ...plugins import SingleIntegrator2DPlugin, Box2DObstacleGeneratorPlugin
    except Exception:
        return None, None, None
    if not config_path.exists():
        return None, None, None
    try:
        config = ExperimentConfig.from_yaml(config_path)
        config.obstacle_levels = [level]
        config.seeds = [seed]
        runner = ExperimentRunner(config)
        runner.register_plugin(SingleIntegrator2DPlugin(), "environment")
        runner.register_plugin(Box2DObstacleGeneratorPlugin(), "obstacle_generator")
        np.random.seed(seed)
        env_plugin = runner.registry.get_plugin("environment", config.env_name)
        temp_env = env_plugin.create_env(config.env_params)
        target_pos = np.asarray(temp_env.target, dtype=np.float32)
        start_pos = runner._generate_start_position(level, seed, temp_env, env_plugin)
        obstacle_gen = runner.registry.get_plugin("obstacle_generator", config.obstacle_config.get("generator", "box2d"))
        obstacle_config_with_env = {**config.obstacle_config, "env_name": config.env_name}
        obstacles = obstacle_gen.generate(level, seed, start_pos, target_pos, obstacle_config_with_env)
        start_2d = np.asarray(start_pos, dtype=np.float64).flatten()[:2]
        goal_2d = np.asarray(target_pos, dtype=np.float64).flatten()[:2]
        return start_2d, goal_2d, obstacles
    except Exception:
        return None, None, None


# ---------------------------------------------------------------------------
# Plot 2: Best trajectories per (level, seed), all algos; with obstacles
# ---------------------------------------------------------------------------
def plot_best_trajectories(
    results_root: Path,
    out_dir: Path,
    config_path: Path | None = None,
) -> None:
    """For each (level, seed), plot best trajectory of each algorithm on one figure with obstacles.
    If config_path is set and exists, obstacles are generated via the same logic as env_preview."""
    subdir = out_dir / "best_trajectories"
    subdir.mkdir(parents=True, exist_ok=True)

    use_runner_obstacles = config_path is not None and config_path.exists()

    for level in LEVELS:
        if not use_runner_obstacles:
            config = {**BOX2D_CONFIG}
            config["enable_nonconvexity_check"] = level in (7, 8, 9)

        for seed in SEEDS:
            if use_runner_obstacles:
                start_pos, target_pos, obstacles = _get_env_setup_from_config(config_path, level, seed)
                if start_pos is None or target_pos is None:
                    continue
                map_bounds = BOX2D_CONFIG["map_bounds"]
            else:
                start_pos = None
                target_pos = None
                for algo in ALGOS_ORDER:
                    traj_path = results_root / algo / f"level_{level}" / f"seed_{seed}" / "trajectory" / "trajectory.json"
                    if traj_path.exists():
                        with open(traj_path) as f:
                            data = json.load(f)
                        cand = data.get("candidate_states", [])
                        best_idx = data.get("best_idx", 0)
                        if cand and 0 <= best_idx < len(cand):
                            states = np.asarray(cand[best_idx])
                            if states.ndim == 2 and len(states) >= 2:
                                start_pos = np.asarray(states[0], dtype=np.float64).reshape(2)
                                target_pos = np.asarray(states[-1], dtype=np.float64).reshape(2)
                        break
                if start_pos is None or target_pos is None:
                    continue
                try:
                    obstacles = generate_box2d_obstacles(level, seed, start_pos, target_pos, config)
                except Exception:
                    obstacles = None
                map_bounds = config.get("map_bounds", BOX2D_CONFIG["map_bounds"])

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.set_aspect("equal")
            mb = map_bounds if map_bounds else BOX2D_CONFIG["map_bounds"]
            ax.set_xlim(mb.get("x_min", -1.5), mb.get("x_max", 1.0))
            ax.set_ylim(mb.get("y_min", -2.0), mb.get("y_max", 0.5))
            if obstacles is not None:
                draw_obstacles(ax, obstacles)

            for algo in ALGOS_ORDER:
                traj_path = results_root / algo / f"level_{level}" / f"seed_{seed}" / "trajectory" / "trajectory.json"
                if not traj_path.exists():
                    continue
                with open(traj_path) as f:
                    data = json.load(f)
                cand = data.get("candidate_states", [])
                best_idx = data.get("best_idx", 0)
                if not cand or best_idx < 0 or best_idx >= len(cand):
                    continue
                states = np.asarray(cand[best_idx])
                if states.ndim == 1:
                    continue
                if states.shape[1] > 2:
                    states = states[:, :2]
                color = _algo_color(algo)
                ax.plot(states[:, 0], states[:, 1], color=color, linewidth=2.0, label=_algo_display_name(algo))

            ax.plot(start_pos[0], start_pos[1], "ko", markersize=8, label="Start")
            ax.plot(target_pos[0], target_pos[1], "r*", markersize=15, label="Target")
            ax.set_title(f"Best Trajectories Level {level} Seed {seed}")
            ax.legend(loc="best", fontsize=8)
            ax.set_facecolor("white")
            ax.grid(True, alpha=0.3)
            fig.savefig(subdir / f"level_{level}_seed_{seed}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        print(f"  best_trajectories: level_{level} (10 seeds)")

    print(f"  best_trajectories: 100 figures in {subdir}")


# ---------------------------------------------------------------------------
# Plot 3: CFS-MBD adaptive metrics (3 algos, 11 subplots per level)
# ---------------------------------------------------------------------------
def plot_cfsmbd_adaptive(results_root: Path, out_dir: Path) -> None:
    """For each level, plot 11 adaptive metrics; each subplot has 3 algos with mean ± SEM."""
    subdir = out_dir / "cfsmbd_adaptive"
    subdir.mkdir(parents=True, exist_ok=True)

    metric_names = ["r_k", "v_rate", "v_mean", "c_k", "nu", "p_k", "rho", "topK", "I_QP", "eps", "lambda"]
    integer_metrics = {"topK", "I_QP"}

    for level in LEVELS:
        # Collect (K,) mean and SEM per algo over seeds (each seed: mean over modes first)
        algo_curves: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {a: [] for a in CFSMBD_ALGOS}
        K_ref = None

        for algo in CFSMBD_ALGOS:
            seed_curves = []  # list of (K, 11) arrays
            for seed in SEEDS:
                metrics_path = (
                    results_root / algo / f"level_{level}" / f"seed_{seed}" / "adaptive" / "metrics.json"
                )
                if not metrics_path.exists():
                    continue
                with open(metrics_path) as f:
                    data = json.load(f)
                raw = data.get("metrics", [])
                names = data.get("metric_names", metric_names)
                if not raw:
                    continue
                # raw can be nested list (C, K, 11) or (K, 11); null (c_k/nu) → nan for plotting
                arr = _adaptive_metrics_raw_to_array(raw)
                if arr.ndim == 2:
                    arr = np.expand_dims(arr, axis=0)
                if arr.ndim != 3:
                    continue
                C, K, _ = arr.shape
                seed_mean = np.nanmean(arr, axis=0)  # (K, 11)
                seed_curves.append(seed_mean)
                if K_ref is None:
                    K_ref = K

            if not seed_curves:
                continue
            stack = np.array(seed_curves)  # (n_seeds, K, 11)
            n_s = stack.shape[0]
            for j in range(11):
                vals = stack[:, :, j]  # (n_seeds, K)
                mean_j = np.nanmean(vals, axis=0)
                sem_j = (np.nanstd(vals, axis=0) / np.sqrt(n_s) if n_s > 0 else np.zeros(vals.shape[1])) * UNCERTAINTY_MULTIPLIER
                sem_j = np.where(np.isnan(sem_j), 0.0, sem_j)
                algo_curves[algo].append((mean_j, sem_j))

        if K_ref is None:
            continue

        n_metrics = 11
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.atleast_2d(axes)
        for idx in range(n_metrics):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            x = np.arange(K_ref)
            for algo in CFSMBD_ALGOS:
                if algo not in algo_curves or len(algo_curves[algo]) <= idx:
                    continue
                mean_j, sem_j = algo_curves[algo][idx]
                if len(mean_j) != K_ref:
                    mean_j = np.resize(mean_j, K_ref)
                    sem_j = np.resize(sem_j, K_ref)
                if metric_names[idx] in integer_metrics:
                    mean_j = np.round(mean_j)
                color = _algo_color(algo)
                ax.fill_between(x, mean_j - sem_j, mean_j + sem_j, alpha=STD_BAND_ALPHA, color=color)
                ax.plot(x, mean_j, color=color, linewidth=1.5, label=_algo_display_name(algo))
            _set_diffusion_axis(ax, K_ref)
            ax.set_ylabel(_metric_label(metric_names[idx]))
            ax.set_title(_metric_label(metric_names[idx]))
            ax.legend(loc="best", fontsize=7)
            ax.grid(True, alpha=0.3)
            if metric_names[idx] in integer_metrics:
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        for idx in range(n_metrics, axes.size):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)
        fig.suptitle(f"CFS-MBD Adaptive Metrics (Level {level})", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(subdir / f"level_{level}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  cfsmbd_adaptive: level_{level}.png")

    # One summary figure: average over all levels (and seeds)
    metric_names = ["r_k", "v_rate", "v_mean", "c_k", "nu", "p_k", "rho", "topK", "I_QP", "eps", "lambda"]
    integer_metrics = {"topK", "I_QP"}
    algo_all_curves: dict[str, list[np.ndarray]] = {}
    for algo in CFSMBD_ALGOS:
        all_curves = []
        for level in LEVELS:
            for seed in SEEDS:
                metrics_path = (
                    results_root / algo / f"level_{level}" / f"seed_{seed}" / "adaptive" / "metrics.json"
                )
                if not metrics_path.exists():
                    continue
                with open(metrics_path) as f:
                    data = json.load(f)
                raw = data.get("metrics", [])
                if not raw:
                    continue
                arr = _adaptive_metrics_raw_to_array(raw)
                if arr.ndim == 2:
                    arr = np.expand_dims(arr, axis=0)
                if arr.ndim != 3:
                    continue
                seed_mean = np.nanmean(arr, axis=0)  # (K, 11)
                all_curves.append(seed_mean)
        if all_curves:
            algo_all_curves[algo] = all_curves
    K_ref = 100
    if algo_all_curves:
        K_ref = min(c.shape[0] for curves in algo_all_curves.values() for c in curves)
    algo_curves: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {a: [] for a in CFSMBD_ALGOS}
    for algo, all_curves in algo_all_curves.items():
        stack = np.array([c[:K_ref] for c in all_curves])  # (N, K, 11)
        n_s = stack.shape[0]
        for j in range(11):
            vals = stack[:, :, j]
            mean_j = np.nanmean(vals, axis=0)
            sem_j = (np.nanstd(vals, axis=0) / np.sqrt(n_s) if n_s > 0 else np.zeros(vals.shape[1])) * UNCERTAINTY_MULTIPLIER
            sem_j = np.where(np.isnan(sem_j), 0.0, sem_j)
            algo_curves[algo].append((mean_j, sem_j))
    if K_ref is not None and any(len(algo_curves[a]) == 11 for a in CFSMBD_ALGOS):
        n_metrics = 11
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.atleast_2d(axes)
        for idx in range(n_metrics):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            x = np.arange(K_ref)
            for algo in CFSMBD_ALGOS:
                if algo not in algo_curves or len(algo_curves[algo]) <= idx:
                    continue
                mean_j, sem_j = algo_curves[algo][idx]
                if metric_names[idx] in integer_metrics:
                    mean_j = np.round(mean_j)
                color = _algo_color(algo)
                ax.fill_between(x, mean_j - sem_j, mean_j + sem_j, alpha=STD_BAND_ALPHA, color=color)
                ax.plot(x, mean_j, color=color, linewidth=1.5, label=_algo_display_name(algo))
            _set_diffusion_axis(ax, K_ref)
            ax.set_ylabel(_metric_label(metric_names[idx]))
            ax.set_title(_metric_label(metric_names[idx]))
            ax.legend(loc="best", fontsize=7)
            ax.grid(True, alpha=0.3)
            if metric_names[idx] in integer_metrics:
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        for idx in range(n_metrics, axes.size):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)
        fig.suptitle(r"CFS-MBD Adaptive Metrics (All Levels Mean $\pm$ SEM)", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(subdir / "all_levels_mean.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  cfsmbd_adaptive: all_levels_mean.png")

        # Save each metric as a separate figure
        for idx in range(n_metrics):
            fig_one, ax_one = plt.subplots(1, 1, figsize=(8, 3))
            x = np.arange(K_ref)
            for algo in CFSMBD_ALGOS:
                if algo not in algo_curves or len(algo_curves[algo]) <= idx:
                    continue
                mean_j, sem_j = algo_curves[algo][idx]
                if metric_names[idx] in integer_metrics:
                    mean_j = np.round(mean_j)
                color = _algo_color(algo)
                ax_one.fill_between(x, mean_j - sem_j, mean_j + sem_j, alpha=STD_BAND_ALPHA, color=color)
                ax_one.plot(x, mean_j, color=color, linewidth=3.0, label=_algo_display_name(algo))
            if metric_names[idx] in ['eps', 'rho']:
                _set_diffusion_axis(ax_one, K_ref)
            else:
                ax_one.set_xticks([])

            if metric_names[idx] in ['v_mean']:
                ax_one.set_ylabel(_metric_label('v'), fontsize=18)
            elif metric_names[idx] in ['r_k']:
                ax_one.set_ylabel(_metric_label('r'), fontsize=18)
            elif metric_names[idx] in ['c_k']:
                ax_one.set_ylabel(_metric_label('c'), fontsize=18)
            elif metric_names[idx] in ['p_k']:
                ax_one.set_ylabel(_metric_label('p'), fontsize=18)
            elif metric_names[idx] in ['rho']:
                ax_one.set_ylabel(_metric_label('rho'), fontsize=18)
            elif metric_names[idx] in ['topK']:
                ax_one.set_ylabel(_metric_label('H'), fontsize=18)
            elif metric_names[idx] in ['I_QP']:
                ax_one.set_ylabel(_metric_label('I'), fontsize=18)
            elif metric_names[idx] in ['eps']:
                ax_one.set_ylabel(_metric_label('varepsilon'), fontsize=18)
            else:
                ax_one.set_ylabel(_metric_label(metric_names[idx]), fontsize=18)
            ax_one.tick_params(axis="y", labelsize=20)
            # ax_one.set_title(_metric_label(metric_names[idx]), fontsize=21, fontweight="bold")
            # ax_one.legend(loc="best", fontsize=15)
            # ax_one.grid(True, alpha=0.3)
            if metric_names[idx] in integer_metrics:
                ax_one.yaxis.set_major_locator(MaxNLocator(integer=True))
            fig_one.tight_layout()
            safe_name = metric_names[idx].replace("/", "_")
            fig_one.savefig(subdir / f"all_levels_mean_{safe_name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig_one)
        print(f"  cfsmbd_adaptive: all_levels_mean_<metric>.png (11 single-metric figures)")


# ---------------------------------------------------------------------------
# Plot 4: SSR heatmap (6 algos x 10 levels)
# ---------------------------------------------------------------------------
def plot_ssr_heatmap(results_root: Path, out_dir: Path) -> None:
    """Heatmap: rows = algos, cols = levels; color green (1) to red (0)."""
    subdir = out_dir
    subdir.mkdir(parents=True, exist_ok=True)

    data = np.full((len(ALGOS_ORDER), len(LEVELS)), np.nan)
    for i, algo in enumerate(ALGOS_ORDER):
        for j, level in enumerate(LEVELS):
            ssrs = []
            for seed in SEEDS:
                res_path = results_root / algo / f"level_{level}" / f"seed_{seed}" / "results.json"
                if not res_path.exists():
                    continue
                with open(res_path) as f:
                    r = json.load(f)
                m = r.get("metrics", {}) or {}
                ssr_obj = m.get("ssr", {})
                if isinstance(ssr_obj, dict) and "ssr" in ssr_obj:
                    ssrs.append(float(ssr_obj["ssr"]))
            if ssrs:
                data[i, j] = np.mean(ssrs)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # 0 -> red, 1 -> green
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_yticks(range(len(ALGOS_ORDER)))
    ax.set_yticklabels([_algo_display_name(a) for a in ALGOS_ORDER], fontsize=15)
    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels([f"L{l}" for l in LEVELS], rotation=45, ha="right", rotation_mode="anchor", fontsize=12)
    # Region labels: L1-L3 Easy, L4-L6 Constrained, L7-L10 Union; vertical dividers between regions
    ax.axvline(x=2.5, color="black", linestyle="--", linewidth=3, alpha=0.7)
    ax.axvline(x=5.5, color="black", linestyle="--", linewidth=3, alpha=0.7)
    ax.annotate("Easy Maps", xy=(1.0 / len(LEVELS), -0.18), xycoords="axes fraction", ha="center", fontsize=15, color="black")
    ax.annotate("Constrained Maps", xy=(4.5 / len(LEVELS), -0.18), xycoords="axes fraction", ha="center", fontsize=15, color="black")
    ax.annotate("Union Maps", xy=(8.5 / len(LEVELS), -0.18), xycoords="axes fraction", ha="center", fontsize=15, color="black")
    # ax.set_ylabel("Algorithm")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                text = "—"
            else:
                text = f"{val:.2f}"
            color = "white" if not np.isnan(val) and (val < 0.4 or val > 0.6) else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)
    plt.colorbar(im, ax=ax)
    ax.set_title("Safe && Success Rate (SSR) over 10 Seeds", fontsize=22, fontweight="bold")
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(subdir / "ssr_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ssr_heatmap: ssr_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 5: Mean planning time vs level (1 figure: x=level 1..10, y=time, line + SEM band per algo)
# ---------------------------------------------------------------------------
def plot_time_per_level(results_root: Path, out_dir: Path) -> None:
    """One figure: x = level 1..10, y = planning time (s); one line + SEM band per algorithm."""
    subdir = out_dir
    subdir.mkdir(parents=True, exist_ok=True)

    x_levels = np.array(LEVELS, dtype=np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for algo in ALGOS_ORDER:
        means = []
        sems = []
        for level in LEVELS:
            res_dir = results_root / algo / f"level_{level}"
            t_list = []
            if res_dir.exists():
                for seed in SEEDS:
                    res_path = res_dir / f"seed_{seed}" / "results.json"
                    if res_path.exists():
                        with open(res_path) as f:
                            r = json.load(f)
                        pt = r.get("planning_time")
                        if pt is not None:
                            t_list.append(float(pt))
            if t_list:
                n_s = len(t_list)
                means.append(np.mean(t_list))
                sems.append((np.std(t_list) / np.sqrt(n_s) if n_s > 0 else 0.0) * UNCERTAINTY_MULTIPLIER)
            else:
                means.append(np.nan)
                sems.append(np.nan)
        means = np.array(means)
        sems = np.array(sems)
        color = _algo_color(algo)
        ax.fill_between(x_levels, means - sems, means + sems, alpha=STD_BAND_ALPHA, color=color)
        ax.plot(x_levels, means, color=color, linewidth=2.0, label=_algo_display_name(algo))

    ax.set_xticks(LEVELS)
    ax.set_xticklabels([f"L{l}" for l in LEVELS], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Level")
    ax.axvline(x=3.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(x=6.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.annotate("Easy", xy=(1.0 / len(LEVELS), -0.12), xycoords="axes fraction", ha="center", fontsize=9, color="gray")
    ax.annotate("Constrained", xy=(4.5 / len(LEVELS), -0.12), xycoords="axes fraction", ha="center", fontsize=9, color="gray")
    ax.annotate("Union", xy=(8.5 / len(LEVELS), -0.12), xycoords="axes fraction", ha="center", fontsize=9, color="gray")
    ax.set_ylabel("Planning time (s)")
    ax.set_title("Planning Time vs Level (mean ± SEM over 10 seeds)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(subdir / "time_per_level.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  time_per_level: time_per_level.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    default_config = _PROJECT_ROOT / "configs" / "single_2d" / "cfsmbd.yaml"
    parser = argparse.ArgumentParser(description="Single2d visualizations")
    parser.add_argument(
        "--which",
        choices=["cost", "trajectories", "adaptive", "heatmap", "time", "all"],
        default="all",
        help="Which visualization(s) to run",
    )
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT, help="Path to results/single2d")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory (default: results/single2d/visualizations)")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Experiment YAML config for best_trajectories obstacles (same as env_preview). Default: configs/single_2d/cfsmbd.yaml",
    )
    args = parser.parse_args()

    results_root = args.results_root
    out_dir = args.out_dir
    _config_path = args.config if args.config.is_absolute() else _PROJECT_ROOT / args.config
    config_path = _config_path if _config_path.exists() else None
    if not results_root.exists():
        raise SystemExit(f"Results root does not exist: {results_root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    if config_path:
        print(f"Best-trajectory obstacles: using config {config_path} (same as env_preview)")

    which = args.which
    if which in ("cost", "all"):
        print("Plot 1: Cost vs diffusion step...")
        plot_cost_vs_step(results_root, out_dir)
    if which in ("trajectories", "all"):
        print("Plot 2: Best trajectories (with obstacles)...")
        plot_best_trajectories(results_root, out_dir, config_path=config_path)
    if which in ("adaptive", "all"):
        print("Plot 3: CFS-MBD adaptive metrics...")
        plot_cfsmbd_adaptive(results_root, out_dir)
    if which in ("heatmap", "all"):
        print("Plot 4: SSR heatmap...")
        plot_ssr_heatmap(results_root, out_dir)
    if which in ("time", "all"):
        print("Plot 5: Planning time per level...")
        plot_time_per_level(results_root, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
