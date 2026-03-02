"""
Plot trajectory_modes_plan from all seeds on one figure per level (D3IL avoiding).

Each seed uses one color; all modes (candidate trajectories) for that seed are drawn
with the same style (no bold for best). Outputs go to results/d3il_avoiding/visualizations/
like results/single2d/visualizations. Method is passed via CLI (e.g. mdcoas, mdcoas-f).

Usage (from project root):
  python scripts/visualize_d3il_trajectory_all_seeds.py --method mdcoas
  python scripts/visualize_d3il_trajectory_all_seeds.py --method mdcoas-f

Output: results/d3il_avoiding/visualizations/<method>/level_<i>.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Seed colors: distinct, no red (obstacles are red), seed 1 = pink from run_single2d_viz ALGO_COLORS
SEED_COLORS = [
    "#9467bd",   # purple
    "#2ca02c",   # green  # "#CC79A7" pink
    "#d62728",   # red  # "#8c564b" brown
    "#ff7f0e",   # orange
    "#1f77b4",   # blue
]


def extract_position_9d(state: np.ndarray) -> np.ndarray:
    """Extract xy from 9D state [tcp_xy, q] for D3IL avoiding 9D."""
    s = np.asarray(state, dtype=np.float32).reshape(-1)
    if s.size >= 2:
        return s[:2]
    return s[: min(2, s.size)]


def load_trajectory_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_all_modes_xy(traj_data: dict) -> list[np.ndarray]:
    """Return list of (N, 2) positions, one per mode (candidate_states entry)."""
    cand = traj_data.get("candidate_states", [])
    out = []
    for states in cand:
        positions = np.array([extract_position_9d(np.asarray(s)) for s in states], dtype=np.float32)
        out.append(positions)
    return out


def smooth_positions(positions: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Smooth (N, 2) trajectory with 1D Gaussian along the path. Returns same shape."""
    if len(positions) < 3:
        return positions
    try:
        from scipy.ndimage import gaussian_filter1d
        out = np.empty_like(positions)
        out[:, 0] = gaussian_filter1d(positions[:, 0], sigma=sigma, mode="nearest")
        out[:, 1] = gaussian_filter1d(positions[:, 1], sigma=sigma, mode="nearest")
        return out
    except ImportError:
        return positions


def build_obstacles_for_level(level: int, obstacle_config: dict):
    """Build ObstacleManager for a given level using D3IL fixed preset."""
    from genedynamics.experiments.plugins.obstacles.d3il_avoiding_fixed import (
        D3ILAvoidingFixedGeneratorPlugin,
    )

    plugin = D3ILAvoidingFixedGeneratorPlugin()
    start_pos = np.array([0.5, -0.28], dtype=np.float32)
    target_pos = np.array([0.5, 0.35], dtype=np.float32)
    return plugin.generate(level, 0, start_pos, target_pos, obstacle_config)


def draw_obstacles_and_target(ax, obstacles, map_bounds, use_target_line: bool = True):
    """Draw obstacles and target line on ax (D3IL style)."""
    from genedynamics.experiments.common.visualization import (
        draw_obstacles,
        D3IL_BG_YELLOW,
        D3IL_OBSTACLE_RED,
        D3IL_TARGET_GREEN,
    )
    from genedynamics.experiments.plugins.obstacles.d3il_avoiding_fixed import D3IL_TARGET_LINE_Y

    x_min = map_bounds.get("x_min", 0.2)
    x_max = map_bounds.get("x_max", 0.8)
    y_min = map_bounds.get("y_min", -0.3)
    y_max = map_bounds.get("y_max", 0.4)

    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_facecolor(D3IL_BG_YELLOW)
    draw_obstacles(ax, obstacles, obstacle_color=D3IL_OBSTACLE_RED, obstacle_alpha=1.0)

    if use_target_line:
        xs = np.array([x_min, x_max])
        ax.plot(
            xs,
            np.full_like(xs, D3IL_TARGET_LINE_Y),
            color=D3IL_TARGET_GREEN,
            linewidth=5.0,
            linestyle="-",
            zorder=10,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.4")
        spine.set_linewidth(0.8)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(
        description="Plot trajectory_modes_plan from all seeds per level (one figure per level)."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="mdcoas",
        help="Method name (e.g. mdcoas, mdcoas-f, mdoc). Used for results dir and config.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Override results root (default: results/d3il_avoiding/<method>).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment YAML (default: configs/d3il_avoiding/<method>.yaml).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Override visualization output root (default: results/d3il_avoiding/visualizations/<method>).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if args.config is None:
        config_path = project_root / "configs" / "d3il_avoiding" / f"{args.method}.yaml"
    else:
        config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    from genedynamics.experiments.framework import ExperimentConfig

    config = ExperimentConfig.from_yaml(config_path)
    obstacle_config = getattr(config, "obstacle_config", None) or {}
    map_bounds = obstacle_config.get("map_bounds", {})
    method_params = getattr(config, "method_params", None) or {}
    use_target_line = bool(method_params.get("use_target_line", True))

    if args.results_dir is not None:
        results_root = Path(args.results_dir)
    else:
        results_root = config.output_dir if config.output_dir.is_absolute() else (project_root / config.output_dir)
    results_root = results_root.resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"Results dir not found: {results_root}")

    # Output under results/d3il_avoiding/visualizations/<method>/ (like results/single2d/visualizations)
    if args.out_dir is not None:
        viz_root = Path(args.out_dir)
    else:
        viz_root = project_root / "results" / "d3il_avoiding" / "visualizations" / args.method
    viz_root.mkdir(parents=True, exist_ok=True)

    level_dirs = sorted([d for d in results_root.iterdir() if d.is_dir() and d.name.startswith("level_")])
    if not level_dirs:
        print(f"No level_* dirs in {results_root}")
        return

    for level_dir in level_dirs:
        level_name = level_dir.name
        try:
            level = int(level_name.replace("level_", ""))
        except ValueError:
            continue

        trajectory_jsons = []
        for seed_dir in sorted(level_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            tj = seed_dir / "trajectory" / "trajectory.json"
            if tj.exists():
                trajectory_jsons.append((seed_dir.name, tj))

        if not trajectory_jsons:
            print(f"  {level_name}: no trajectory/trajectory.json found, skip")
            continue

        obstacles = build_obstacles_for_level(level, obstacle_config)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        draw_obstacles_and_target(ax, obstacles, map_bounds, use_target_line=use_target_line)

        for i, (seed_label, tj_path) in enumerate(trajectory_jsons):
            traj_data = load_trajectory_json(tj_path)
            all_positions = get_all_modes_xy(traj_data)
            color = SEED_COLORS[i % len(SEED_COLORS)]
            legend_label = seed_label.replace("seed_", "seed ")
            for mode_idx, positions in enumerate(all_positions):
                use_label = legend_label if mode_idx == 0 else "_nolegend_"
                if len(positions) >= 2:
                    positions = smooth_positions(positions)
                    ax.plot(
                        positions[:, 0],
                        positions[:, 1],
                        color=color,
                        linewidth=1.5,
                        alpha=0.4,
                        label=use_label,
                        zorder=5,
                    )
                elif len(positions) == 1:
                    ax.scatter(
                        positions[0, 0],
                        positions[0, 1],
                        color=color,
                        s=20,
                        alpha=0.4,
                        label=use_label,
                        zorder=5,
                    )

        # ax.legend(loc="best", fontsize=10)

        out_path = viz_root / f"level_{level}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {level_name}: saved {out_path} ({len(trajectory_jsons)} seeds)")


if __name__ == "__main__":
    main()
