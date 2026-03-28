"""
Stepping-stones trajectory visualizations.
"""

from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from genedynamics.tasks.stepping_stones import decode_plan_states
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
    if bool(getattr(scene, "has_river", True)):
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


def _swing_pair(mode: int):
    m = int(mode) % 4
    if m == 0:
        return ("FR", "RL")
    if m == 2:
        return ("FL", "RR")
    return tuple()


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
        if states.ndim == 2 and states.shape[1] >= 3:
            body, _yaw, feet, mode = decode_plan_states(
                states,
                step_width=float(getattr(env, "step_width", getattr(env, "stance_width", 0.30))),
                half_pair_length=float(getattr(env, "fore_hind_offset", 0.18)),
            )
            foot_radius = _infer_foot_radius(viz_cfg)
            tracks = [
                ("FL", feet["FL"], "#ff7f0e"),
                ("FR", feet["FR"], "#2ca02c"),
                ("RL", feet["RL"], "#9467bd"),
                ("RR", feet["RR"], "#8c564b"),
            ]
            ax.plot(body[:, 0], body[:, 1], color="#1664c0", linewidth=1.8, alpha=0.9, zorder=3, label="body")
            for leg, pts, color in tracks:
                ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.2, alpha=0.80, zorder=2.7, label=leg)
                ax.scatter(pts[:, 0], pts[:, 1], s=8, color=color, alpha=0.18, zorder=2.5)
            for t, mode_idx in enumerate(mode):
                for swing_leg in _swing_pair(int(mode_idx)):
                    p = feet[swing_leg][t]
                    ax.add_patch(
                        plt.Circle(
                            (float(p[0]), float(p[1])),
                            float(1.5 * foot_radius),
                            facecolor="#f04f88",
                            edgecolor="none",
                            alpha=0.08,
                            zorder=2.1,
                        )
                    )
            ax.scatter(body[0, 0], body[0, 1], marker="o", s=28, color="white", edgecolor="#1664c0", zorder=5)
            ax.scatter(body[-1, 0], body[-1, 1], marker="*", s=90, color="#f04f88", zorder=5)
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
        step_width = float(getattr(env, "step_width", getattr(env, "stance_width", 0.30)))
        for i, st in enumerate(cand_states):
            s = np.asarray(st, dtype=np.float32)
            if s.ndim != 2 or s.shape[1] < 3:
                continue
            mid, _yaw, _feet, _phase = decode_plan_states(
                s,
                step_width=step_width,
                half_pair_length=float(getattr(env, "fore_hind_offset", 0.18)),
            )
            lw = 2.6 if i == best_idx else 1.1
            a = 0.95 if i == best_idx else 0.35
            ax.plot(mid[:, 0], mid[:, 1], color=colors[i], linewidth=lw, alpha=a, zorder=3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.25)

    def save(self, output_path: Any, fig: Any, **kwargs: Any) -> None:
        fig.savefig(output_path, dpi=kwargs.get("dpi", 150), bbox_inches=kwargs.get("bbox_inches", "tight"))

