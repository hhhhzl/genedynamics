"""
Stepping-stones trajectory visualizations.
"""

from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from ...framework.base import VisualizationPlugin


def _get_scene(env: Any, obstacles: Any):
    if env is not None and hasattr(env, "scene") and getattr(env, "scene") is not None:
        return getattr(env, "scene")
    if obstacles is not None and hasattr(obstacles, "stepping_scene"):
        return getattr(obstacles, "stepping_scene")
    return None


def _draw_scene(ax: Any, scene: Any):
    ax.set_aspect("equal")
    ax.set_xlim(scene.map_x[0], scene.map_x[1])
    ax.set_ylim(scene.map_y[0], scene.map_y[1])
    ax.set_facecolor("#f7f9fc")
    rx0, rx1 = scene.river_x
    ry0, ry1 = scene.map_y
    ax.add_patch(
        plt.Rectangle(
            (rx0, ry0),
            rx1 - rx0,
            ry1 - ry0,
            facecolor="#d9ecff",
            edgecolor="#7aa6d9",
            linewidth=1.0,
            alpha=0.9,
            zorder=0.5,
        )
    )
    for c, r in zip(np.asarray(scene.stones_centers), np.asarray(scene.stones_radii)):
        ax.add_patch(
            plt.Circle(
                (float(c[0]), float(c[1])),
                float(r),
                facecolor="#9099a2",
                edgecolor="#59636e",
                linewidth=0.7,
                alpha=0.8,
                zorder=1,
            )
        )


def _infer_foot_radius(viz_cfg: Dict[str, Any]) -> float:
    if "foot_radius" in viz_cfg:
        return float(viz_cfg["foot_radius"])
    cfg = viz_cfg.get("config", None)
    if cfg is not None and hasattr(cfg, "obstacle_config"):
        obs_cfg = getattr(cfg, "obstacle_config", {}) or {}
        if "robot_radius" in obs_cfg:
            return float(obs_cfg["robot_radius"])
    return 0.03


class SteppingStonesTrajectoryVisualizationPlugin(VisualizationPlugin):
    @property
    def name(self) -> str:
        return "stepping_trajectory"

    def visualize(self, fig: Any, ax: Any, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        traj = data["trajectory"]
        env = data["env"]
        obstacles = data["obstacles"]
        viz_cfg = config or {}
        scene = _get_scene(env, obstacles)
        if scene is None:
            ax.text(0.5, 0.5, "missing stepping scene", ha="center", va="center", transform=ax.transAxes)
            return
        _draw_scene(ax, scene)

        states = np.asarray([np.asarray(s, dtype=np.float32).reshape(-1) for s in traj.states], dtype=np.float32)
        if states.ndim == 2 and states.shape[1] >= 4:
            p_l = states[:, :2]
            p_r = states[:, 2:4]
            mid = 0.5 * (p_l + p_r)
            draw_mid = bool(viz_cfg.get("draw_midfoot", False))
            draw_virtual_quad = bool(viz_cfg.get("draw_virtual_quadruped", False))
            half_pair_length = float(viz_cfg.get("virtual_pair_half_length", 0.18))
            foot_radius = _infer_foot_radius(viz_cfg)

            # Real task semantics: two optimized trajectories (left/right diagonal pairs).
            ax.plot(p_l[:, 0], p_l[:, 1], color="#d1495b", linewidth=1.4, alpha=0.90, zorder=3, label="left-pair")
            ax.plot(p_r[:, 0], p_r[:, 1], color="#00798c", linewidth=1.4, alpha=0.90, zorder=3, label="right-pair")

            if draw_mid:
                ax.plot(mid[:, 0], mid[:, 1], color="#1664c0", linewidth=1.6, alpha=0.85, zorder=3, label="mid-foot")

            # Optional virtual quadruped overlay (off by default).
            if draw_virtual_quad:
                fl = p_l + np.array([half_pair_length, 0.0], dtype=np.float32)
                hl = p_l - np.array([half_pair_length, 0.0], dtype=np.float32)
                fr = p_r + np.array([half_pair_length, 0.0], dtype=np.float32)
                hr = p_r - np.array([half_pair_length, 0.0], dtype=np.float32)
                tracks = [("FL", fl, "#ff7f0e"), ("FR", fr, "#2ca02c"), ("HL", hl, "#9467bd"), ("HR", hr, "#8c564b")]
                for _, pts, color in tracks:
                    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.1, alpha=0.70, zorder=2.5)
                    for px, py in pts:
                        ax.add_patch(
                            plt.Circle(
                                (float(px), float(py)),
                                float(foot_radius),
                                facecolor=color,
                                edgecolor=color,
                                linewidth=0.6,
                                alpha=0.12,
                                zorder=2.2,
                            )
                        )

            ax.scatter(mid[0, 0], mid[0, 1], marker="o", s=28, color="white", edgecolor="#1664c0", zorder=5)
            ax.scatter(mid[-1, 0], mid[-1, 1], marker="*", s=90, color="#f04f88", zorder=5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.25)

    def save(self, output_path: Any, fig: Any, **kwargs: Any) -> None:
        fig.savefig(output_path, dpi=kwargs.get("dpi", 150), bbox_inches=kwargs.get("bbox_inches", "tight"))


class SteppingStonesModesVisualizationPlugin(VisualizationPlugin):
    @property
    def name(self) -> str:
        return "stepping_modes"

    def visualize(self, fig: Any, ax: Any, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        result = data["result"]
        env = data["env"]
        obstacles = data["obstacles"]
        scene = _get_scene(env, obstacles)
        if scene is None:
            ax.text(0.5, 0.5, "missing stepping scene", ha="center", va="center", transform=ax.transAxes)
            return
        _draw_scene(ax, scene)

        cand_states = result.get("candidate_states", [])
        if cand_states is None or len(cand_states) == 0:
            cand_states = [result.get("states", [])]
        best_idx = int(result.get("best_idx", 0))
        colors = ["#6a7fdb"] * len(cand_states)
        for i, st in enumerate(cand_states):
            s = np.asarray(st, dtype=np.float32)
            if s.ndim != 2 or s.shape[1] < 4:
                continue
            mid = 0.5 * (s[:, :2] + s[:, 2:4])
            lw = 2.6 if i == best_idx else 1.1
            a = 0.95 if i == best_idx else 0.35
            ax.plot(mid[:, 0], mid[:, 1], color=colors[i], linewidth=lw, alpha=a, zorder=3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.25)

    def save(self, output_path: Any, fig: Any, **kwargs: Any) -> None:
        fig.savefig(output_path, dpi=kwargs.get("dpi", 150), bbox_inches=kwargs.get("bbox_inches", "tight"))

