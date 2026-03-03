#!/usr/bin/env python3
"""
Visualize environments for level 0–10 and seed 0–10: start, goal, and obstacles.

Uses the same config and generation logic as the experiment framework so that
what you see matches actual runs. Saves one figure per level (11 subplots for
seeds 0–10) and optionally a summary grid.

Requires: matplotlib, and project deps (see requirements.txt). Run from project root
  or with PYTHONPATH including project root.

Usage:
  python scripts/visualizations/visualize_env_levels_seeds.py [--config CONFIG] [--outdir OUTDIR]
  python scripts/visualizations/visualize_env_levels_seeds.py --levels 0-10 --seeds 0-10   # default
  python scripts/visualizations/visualize_env_levels_seeds.py --levels 0,5,10 --seeds 0,1  # subset
"""

from pathlib import Path
import sys
import argparse

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from genedynamics.experiments.framework import ExperimentRunner, ExperimentConfig
from genedynamics.experiments.plugins import (
    SingleIntegrator2DPlugin,
    Box2DObstacleGeneratorPlugin,
    ObstacleDensityMetricsPlugin,
    NonconvexityMetricsPlugin,
)
from genedynamics.experiments.common.visualization import draw_obstacles


def register_plugins(runner: ExperimentRunner) -> None:
    """Register plugins needed for visualization and metrics."""
    runner.register_plugin(SingleIntegrator2DPlugin(), "environment")
    runner.register_plugin(Box2DObstacleGeneratorPlugin(), "obstacle_generator")
    runner.register_plugin(ObstacleDensityMetricsPlugin(), "metric")
    runner.register_plugin(NonconvexityMetricsPlugin(), "metric")


def get_env_setup(runner: ExperimentRunner, level: int, seed: int):
    """
    Get start position, target position, and obstacles for (level, seed).
    Mirrors the logic in ExperimentRunner.run_single_experiment (steps 2–4).
    """
    config = runner.config
    np.random.seed(seed)

    env_plugin = runner.registry.get_plugin("environment", config.env_name)
    temp_env = env_plugin.create_env(config.env_params)
    target_pos = np.asarray(temp_env.target, dtype=np.float32)
    start_pos = runner._generate_start_position(level, seed, temp_env, env_plugin)

    obstacle_gen_name = config.obstacle_config.get("generator", "box2d")
    obstacle_gen = runner.registry.get_plugin("obstacle_generator", obstacle_gen_name)
    obstacle_config_with_env = {
        **config.obstacle_config,
        "env_name": config.env_name,
    }
    obstacles = obstacle_gen.generate(
        level, seed, start_pos, target_pos, obstacle_config_with_env
    )

    # Start/goal for 2D: first two components
    start_2d = np.asarray(start_pos, dtype=np.float32).flatten()[:2]
    goal_2d = np.asarray(target_pos, dtype=np.float32).flatten()[:2]

    # Compute density (with robot_radius) and nonconvexity
    density_val = 0.0
    nonconv_val = 0.0
    if len(obstacles) > 0:
        obstacle_config = getattr(config, "obstacle_config", None) or {}
        robot_radius = float(obstacle_config.get("robot_radius", 0.05))
        density_plugin = runner.registry.get_plugin("metric", "obstacle_density")
        nonconv_plugin = runner.registry.get_plugin("metric", "nonconvexity")
        env_for_metrics = env_plugin.create_env(config.env_params)
        dens = density_plugin.compute(
            None, env_for_metrics, obstacles, None,
            robot_radius=robot_radius, level=level, obstacle_config=obstacle_config,
        )
        density_val = float(dens.get("obstacle_density", 0.0))
        nc = nonconv_plugin.compute(
            None, env_for_metrics, obstacles, None,
            robot_radius=robot_radius, level=level, obstacle_config=obstacle_config,
        )
        sdf_nc = nc.get("sdf", {})
        nonconv_val = float(sdf_nc.get("score_raw", sdf_nc.get("mean_violation", 0.0)))

    return start_2d, goal_2d, obstacles, density_val, nonconv_val


def plot_one_cell(
    ax, start_2d, goal_2d, obstacles, map_bounds, level: int, seed: int,
    robot_radius: float = 0.05,
    density: float = 0.0,
    nonconvexity: float = 0.0,
):
    """Draw one (level, seed) environment on ax. Start/goal 按真实 robot_radius 画圆，不放大."""
    x_min = float(map_bounds.get("x_min", -2.0))
    x_max = float(map_bounds.get("x_max", 2.0))
    y_min = float(map_bounds.get("y_min", -2.0))
    y_max = float(map_bounds.get("y_max", 2.0))

    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    if obstacles is not None and len(obstacles) > 0:
        draw_obstacles(ax, obstacles)

    # Start: 按数据坐标画半径为 robot_radius 的圆（真实尺度），中心加小点便于辨认
    circle_start = mpatches.Circle(
        (float(start_2d[0]), float(start_2d[1])),
        radius=robot_radius,
        facecolor="green",
        alpha=0.6,
        edgecolor="darkgreen",
        linewidth=1.5,
        zorder=10,
    )
    ax.add_patch(circle_start)
    ax.scatter(
        [start_2d[0]], [start_2d[1]],
        c="darkgreen", s=12, marker="o", zorder=11, label="Start",
    )
    # Goal: 同上，半径为 robot_radius 的圆 + 中心星
    circle_goal = mpatches.Circle(
        (float(goal_2d[0]), float(goal_2d[1])),
        radius=robot_radius,
        facecolor="none",
        edgecolor="red",
        linewidth=1.5,
        linestyle="-",
        zorder=10,
    )
    ax.add_patch(circle_goal)
    ax.plot(
        goal_2d[0], goal_2d[1],
        "r*", markersize=6, zorder=11, label="Goal",
    )

    title = f"seed {seed}\ndens={density:.3f} nc={nonconvexity:.4f}"
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=8)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize level 0–10, seed 0–10: start, goal, obstacles"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "single_2d" / "cfsmbd.yaml"),
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "env_preview"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="0-10",
        help="Level range, e.g. 0-10 or 0,1,2",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0-10",
        help="Seed range, e.g. 0-10 or 0,1,2",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    if "-" in args.levels:
        lo, hi = args.levels.split("-")
        levels = list(range(int(lo), int(hi) + 1))
    else:
        levels = [int(x) for x in args.levels.split(",")]
    if "-" in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(x) for x in args.seeds.split(",")]

    config = ExperimentConfig.from_yaml(config_path)
    config.obstacle_levels = levels
    config.seeds = seeds

    runner = ExperimentRunner(config)
    register_plugins(runner)

    map_bounds = config.obstacle_config.get(
        "map_bounds",
        {"x_min": -1.5, "x_max": 1.0, "y_min": -2.0, "y_max": 0.5},
    )
    robot_radius = float(config.obstacle_config.get("robot_radius", 0.05))
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_seeds = len(seeds)
    n_cols = min(11, n_seeds)
    n_rows = (n_seeds + n_cols - 1) // n_cols
    figsize_per_cell = 2.2
    figsize = (n_cols * figsize_per_cell, n_rows * figsize_per_cell)

    for level in levels:
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize, squeeze=False
        )
        axes_flat = axes.flat
        for i, seed in enumerate(seeds):
            start_2d, goal_2d, obstacles, density_val, nonconv_val = get_env_setup(
                runner, level, seed
            )
            plot_one_cell(
                axes_flat[i], start_2d, goal_2d, obstacles, map_bounds, level, seed,
                robot_radius=robot_radius,
                density=density_val,
                nonconvexity=nonconv_val,
            )
        for j in range(len(seeds), len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle(
            f"Level {level} — dens (w/ r), nc (nonconvexity) | Obstacles (gray)",
            fontsize=12,
        )
        plt.tight_layout()
        out_file = out_dir / f"level_{level}.png"
        plt.savefig(out_file, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_file}")

    print(f"Done. Figures in {out_dir}")


if __name__ == "__main__":
    main()
