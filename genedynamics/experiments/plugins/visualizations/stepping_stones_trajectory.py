"""
Stepping-stones trajectory visualizations.
"""

from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from genedynamics.envs.obstacles.stepping_stones import foot_stepping_violation_np
from genedynamics.tasks.stepping_stones import decode_plan_states
from ...framework.base import VisualizationPlugin


def _get_scene(env: Any, obstacles: Any):
    if env is not None and hasattr(env, "scene") and getattr(env, "scene") is not None:
        return getattr(env, "scene")
    if obstacles is not None and hasattr(obstacles, "stepping_scene"):
        return getattr(obstacles, "stepping_scene")
    return None


def _is_corridor_scene(scene: Any) -> bool:
    return hasattr(scene, "corridor_width") and hasattr(scene, "obstacles")


def _draw_corridor_scene_shared(ax: Any, scene: Any) -> None:
    """Draw corridor scene (walls + box/sphere obstacles) for shared viz."""
    hw = scene.corridor_width / 2.0
    length = scene.corridor_length
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, length + 0.3)
    ax.set_ylim(-hw - 0.3, hw + 0.3)
    ax.set_facecolor("#f7f9fc")
    # Floor
    ax.add_patch(plt.Rectangle((0, -hw), length, 2 * hw,
        facecolor="#e2e8f0", edgecolor="none", zorder=0))
    # Walls
    for wy in (-hw - 0.08, hw):
        ax.add_patch(plt.Rectangle((0, wy), length, 0.08,
            facecolor="#2d3748", edgecolor="#1a202c", linewidth=0.8, zorder=5))
    # Obstacles
    _z_color = {"full": ("#4a5568", 0.85), "low": ("#e53e3e", 0.55),
                "floor": ("#dd6b20", 0.55), "mid": ("#805ad5", 0.55)}
    for obs in scene.obstacles:
        zlo, zhi = obs.z_min, obs.z_max
        if zlo <= 0.05 and zhi >= 1.8:
            ot = "full"
        elif zlo > 0.4:
            ot = "low"
        elif zhi < 0.5:
            ot = "floor"
        else:
            ot = "mid"
        fc, alpha = _z_color[ot]
        if getattr(obs, "shape", "box") == "sphere":
            ax.add_patch(plt.Circle((obs.cx, obs.cy), obs.radius,
                facecolor=fc, edgecolor="#1a202c", linewidth=0.6,
                alpha=alpha, zorder=3))
        else:
            ax.add_patch(plt.Rectangle(
                (obs.x_min, obs.y_min), obs.x_max - obs.x_min, obs.y_max - obs.y_min,
                facecolor=fc, edgecolor="#1a202c", linewidth=0.6,
                alpha=alpha, zorder=3))
    # Start / Goal
    sx, sy = scene.start_pos
    gx, gy = scene.goal_pos
    ax.plot(sx, sy, "o", color="#38a169", markersize=8, zorder=10)
    ax.plot(gx, gy, "*", color="#e53e3e", markersize=12, zorder=10)


def _draw_scene(ax: Any, scene: Any):
    # Corridor scene dispatch.
    if _is_corridor_scene(scene):
        _draw_corridor_scene_shared(ax, scene)
        return

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
    for row in np.asarray(getattr(scene, "support_platforms", np.zeros((0, 4))), dtype=np.float32).reshape(-1, 4):
        xmin, xmax, ymin, ymax = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                facecolor="#8b949e",
                edgecolor="#2f3640",
                linewidth=1.4,
                alpha=0.88,
                zorder=0.85,
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


def _on_platform_raw(p: np.ndarray, platforms: np.ndarray) -> bool:
    """True if point is inside any support platform (raw bounds, no margin shrink)."""
    for k in range(platforms.shape[0]):
        xmin, xmax, ymin, ymax = float(platforms[k, 0]), float(platforms[k, 1]), float(platforms[k, 2]), float(platforms[k, 3])
        if xmin <= float(p[0]) <= xmax and ymin <= float(p[1]) <= ymax:
            return True
    return False


def _on_any_stone_raw(p: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> bool:
    """True if foot center is within any stone disk (raw radius, no margin)."""
    if centers.shape[0] == 0:
        return False
    d = np.linalg.norm(centers - p[None, :], axis=-1)
    return bool(np.any(d <= radii))


def _swing_pair(mode: int):
    if int(mode) % 2 == 0:
        return ("FR", "RL")
    return ("FL", "RR")


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

        partial_t = data.get("partial_until_step", None)
        gif_style = data.get("gif_style", False)

        states = np.asarray([np.asarray(s, dtype=np.float32).reshape(-1) for s in traj.states], dtype=np.float32)
        if states.ndim == 2 and states.shape[1] >= 3:
            body, _yaw, feet, mode = decode_plan_states(
                states,
                step_width=float(getattr(env, "step_width", getattr(env, "stance_width", 0.30))),
                half_pair_length=float(getattr(env, "fore_hind_offset", 0.18)),
                centerline_y=float(getattr(env, "centerline_y", 0.0)),
                env=env,
            )
            # Determine visible range for GIF partial rendering
            T_all = body.shape[0]
            if partial_t is not None:
                T_vis = min(int(partial_t) + 1, T_all)
            else:
                T_vis = T_all

            foot_radius = _infer_foot_radius(viz_cfg)
            tracks = [
                ("FL", feet["FL"], "#ff7f0e"),
                ("FR", feet["FR"], "#2ca02c"),
                ("RL", feet["RL"], "#9467bd"),
                ("RR", feet["RR"], "#8c564b"),
            ]
            centers_v = np.asarray(scene.stones_centers, dtype=np.float32)
            radii_v = np.asarray(scene.stones_radii, dtype=np.float32)
            plat_v = np.asarray(getattr(scene, "support_platforms", np.zeros((0, 4))), dtype=np.float32)
            sm_v = float(getattr(env, "stone_margin", 0.032))

            # GIF mode: draw full path as ghost, then partial as solid
            if gif_style and partial_t is not None:
                ax.plot(body[:, 0], body[:, 1], color="#1664c0", linewidth=1.0, alpha=0.20, zorder=2.5)
                for _leg, pts_full, color in tracks:
                    ax.plot(pts_full[:, 0], pts_full[:, 1], color=color, linewidth=0.6, alpha=0.12, zorder=2.3)

            ax.plot(body[:T_vis, 0], body[:T_vis, 1], color="#1664c0", linewidth=1.8, alpha=0.9, zorder=3, label="body")
            for leg, pts_full, color in tracks:
                pts = pts_full[:T_vis]
                ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.2, alpha=0.80, zorder=2.7, label=leg)
                ax.scatter(pts[:, 0], pts[:, 1], s=8, color=color, alpha=0.18, zorder=2.5)
                # Red-dot criteria: only flag feet whose center is truly
                # in the gap (unstable).  Skip platform feet and feet whose
                # center physically sits on a stone surface (within raw radius).
                for ti in range(int(pts.shape[0])):
                    pt = pts[ti]
                    if plat_v.shape[0] > 0 and _on_platform_raw(pt, plat_v):
                        continue
                    if _on_any_stone_raw(pt, centers_v, radii_v):
                        continue
                    if foot_stepping_violation_np(pt, centers_v, radii_v, plat_v, sm_v) > 1e-3:
                        ax.scatter(
                            float(pt[0]),
                            float(pt[1]),
                            s=42,
                            facecolors="#d62728",
                            edgecolors="#6a0000",
                            linewidths=0.65,
                            marker="o",
                            zorder=4.2,
                        )
            for t in range(T_vis):
                mode_idx = mode[t]
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
            if gif_style and partial_t is not None and T_vis > 0:
                # Moving body marker at current step
                t_cur = T_vis - 1
                ax.scatter(body[t_cur, 0], body[t_cur, 1], marker="o", s=50, color="#1664c0", edgecolor="white", linewidths=1.0, zorder=6)
                for _leg, pts_full, color in tracks:
                    ax.scatter(pts_full[t_cur, 0], pts_full[t_cur, 1], s=30, color=color, edgecolor="white", linewidths=0.5, zorder=5.5)
            else:
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

        partial_t = data.get("partial_until_step", None)

        cand_states = result.get("candidate_states", [])
        if cand_states is None or len(cand_states) == 0:
            cand_states = [result.get("states", [])]
        best_idx = int(result.get("best_idx", 0))
        colors = ["#6a7fdb"] * len(cand_states)
        foot_colors = {"FL": "#ff7f0e", "FR": "#2ca02c", "RL": "#9467bd", "RR": "#8c564b"}
        step_width = float(getattr(env, "step_width", getattr(env, "stance_width", 0.30)))
        for i, st in enumerate(cand_states):
            s = np.asarray(st, dtype=np.float32)
            if s.ndim != 2 or s.shape[1] < 3:
                continue
            mid, _yaw, feet_i, _phase = decode_plan_states(
                s,
                step_width=step_width,
                half_pair_length=float(getattr(env, "fore_hind_offset", 0.18)),
                centerline_y=float(getattr(env, "centerline_y", 0.0)),
                env=env,
            )
            T_vis = min(int(partial_t) + 1, mid.shape[0]) if partial_t is not None else mid.shape[0]
            is_best = i == best_idx
            lw = 2.6 if is_best else 1.1
            a = 0.95 if is_best else 0.35
            ax.plot(mid[:T_vis, 0], mid[:T_vis, 1], color=colors[i], linewidth=lw, alpha=a, zorder=3)
            for leg in ("FL", "FR", "RL", "RR"):
                fp = feet_i[leg][:T_vis]
                f_lw = 1.0 if is_best else 0.5
                f_a = 0.70 if is_best else 0.18
                ax.plot(fp[:, 0], fp[:, 1], color=foot_colors[leg], linewidth=f_lw, alpha=f_a, zorder=2.7 if is_best else 2.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.25)

    def save(self, output_path: Any, fig: Any, **kwargs: Any) -> None:
        fig.savefig(output_path, dpi=kwargs.get("dpi", 150), bbox_inches=kwargs.get("bbox_inches", "tight"))

