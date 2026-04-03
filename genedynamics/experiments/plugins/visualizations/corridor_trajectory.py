"""
Corridor obstacle avoidance trajectory visualisation.

Renders a top-down 2D view of the corridor with walls, obstacles (shaded
by z-range type), and the planned trajectory showing torso ellipse +
arm positions over time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from ...framework.base import VisualizationPlugin

# Obstacle type colours by z-range.
_Z_COLOR = {
    "full":  ("#4a5568", 0.85),   # full-height wall/protrusion
    "low":   ("#e53e3e", 0.55),   # overhead-only (low bar)
    "floor": ("#dd6b20", 0.55),   # ground-only
    "mid":   ("#805ad5", 0.55),   # mid-height aerial
}


def _obs_type(z_lo: float, z_hi: float) -> str:
    if z_lo <= 0.05 and z_hi >= 1.8:
        return "full"
    if z_lo > 0.4:
        return "low"
    if z_hi < 0.5:
        return "floor"
    return "mid"


def _draw_corridor_scene(ax: plt.Axes, env: Any) -> None:
    scene = env.scene
    hw = scene.corridor_width / 2.0
    length = scene.corridor_length

    ax.set_facecolor("#f7f9fc")
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, length + 0.3)
    ax.set_ylim(-hw - 0.3, hw + 0.3)

    # Corridor floor
    ax.add_patch(plt.Rectangle(
        (0, -hw), length, 2 * hw,
        facecolor="#e2e8f0", edgecolor="none", zorder=0,
    ))

    # Walls
    wall_w = 0.08
    for wy in (-hw - wall_w, hw):
        ax.add_patch(plt.Rectangle(
            (0, wy), length, wall_w,
            facecolor="#2d3748", edgecolor="#1a202c", linewidth=0.8, zorder=5,
        ))

    # Obstacles
    for obs in scene.obstacles:
        otype = _obs_type(obs.z_min, obs.z_max)
        fc, alpha = _Z_COLOR[otype]
        if getattr(obs, "shape", "box") == "sphere":
            ax.add_patch(plt.Circle(
                (obs.cx, obs.cy), obs.radius,
                facecolor=fc, edgecolor="#1a202c", linewidth=0.6,
                alpha=alpha, zorder=3,
            ))
        else:
            w = obs.x_max - obs.x_min
            h = obs.y_max - obs.y_min
            ax.add_patch(plt.Rectangle(
                (obs.x_min, obs.y_min), w, h,
                facecolor=fc, edgecolor="#1a202c", linewidth=0.6,
                alpha=alpha, zorder=3,
            ))
        cx = 0.5 * (obs.x_min + obs.x_max)
        cy = 0.5 * (obs.y_min + obs.y_max)
        w_obs = obs.x_max - obs.x_min
        h_obs = obs.y_max - obs.y_min
        if w_obs > 0.15 and h_obs > 0.06:
            ax.text(cx, cy, obs.name, fontsize=4, ha="center", va="center",
                    color="white", zorder=4, clip_on=True)

    # Start / Goal markers
    sx, sy = scene.start_pos
    gx, gy = scene.goal_pos
    ax.plot(sx, sy, "o", color="#38a169", markersize=10, zorder=10, label="start")
    ax.plot(gx, gy, "*", color="#e53e3e", markersize=14, zorder=10, label="goal")


def _draw_body_ghost(
    ax: plt.Axes,
    x: float, y: float, psi: float, h: float, psi_torso: float,
    a_L: float, a_R: float,
    alpha: float = 0.4,
    zorder: float = 8,
) -> None:
    """Draw a single body ghost: torso ellipse + arm line segments."""
    from genedynamics.envs.humanoid_corridor_2d import (
        TORSO_A, TORSO_B, TORSO_CROUCH_EXTRA, H_NOMINAL, H_MIN, H_MAX,
        ARM_REACH_OPEN, ARM_REACH_TUCKED,
    )
    heading = psi + psi_torso
    a_eff = TORSO_A + TORSO_CROUCH_EXTRA * max(0.0, H_NOMINAL - h)
    # Ellipse color encodes body height: blue=low (crouching), red=high (standing).
    h_norm = np.clip((h - H_MIN) / max(H_MAX - H_MIN, 1e-6), 0.0, 1.0)
    import matplotlib.cm as cm
    h_color = cm.coolwarm_r(h_norm)
    ellipse = mpatches.Ellipse(
        (x, y), width=2 * TORSO_B, height=2 * a_eff,
        angle=np.degrees(heading),
        facecolor=h_color, edgecolor="#2c5282",
        alpha=alpha, linewidth=0.6, zorder=zorder,
    )
    ax.add_patch(ellipse)

    c, s = np.cos(heading), np.sin(heading)
    for sign, a_tuck, color in [(1.0, a_L, "#e53e3e"), (-1.0, a_R, "#dd6b20")]:
        reach = ARM_REACH_OPEN + (ARM_REACH_TUCKED - ARM_REACH_OPEN) * np.clip(a_tuck, 0, 1)
        # Minimum display reach so tucked arms are still visible.
        display_reach = max(reach, 0.12)
        lx, ly = 0.0, sign * display_reach
        wx = x + c * lx - s * ly
        wy = y + s * lx + c * ly
        # Thicker line when arm is tucked to show tuck state.
        lw = 2.0 if a_tuck > 0.5 else 1.2
        ax.plot([x, wx], [y, wy], color=color, linewidth=lw,
                alpha=alpha * 0.9, zorder=zorder + 0.1)
        # Larger dot when tucked.
        ms = 4.0 if a_tuck > 0.5 else 2.5
        ax.plot(wx, wy, "o", color=color, markersize=ms,
                alpha=alpha * 0.9, zorder=zorder + 0.2)


class CorridorTrajectoryVisualizationPlugin(VisualizationPlugin):
    @property
    def name(self) -> str:
        return "corridor_trajectory"

    def visualize(self, fig: Any, ax: Any, data: Dict[str, Any],
                  config: Dict[str, Any]) -> None:
        env = data.get("env")
        traj = data.get("trajectory")
        if env is None or traj is None:
            ax.text(0.5, 0.5, "missing env/trajectory",
                    ha="center", va="center", transform=ax.transAxes)
            return

        if not hasattr(env, "scene"):
            ax.text(0.5, 0.5, "env has no corridor scene",
                    ha="center", va="center", transform=ax.transAxes)
            return

        states = np.asarray(
            [np.asarray(s, dtype=np.float32).ravel() for s in traj.states],
            dtype=np.float32,
        )
        if states.ndim != 2 or states.shape[1] < 14:
            ax.text(0.5, 0.5, "invalid state dim",
                    ha="center", va="center", transform=ax.transAxes)
            return

        T = states.shape[0]
        _draw_corridor_scene(ax, env)

        # Trajectory line.
        ax.plot(states[:, 0], states[:, 1], "-", color="#3182ce",
                linewidth=1.8, alpha=0.5, zorder=6, label="trajectory")

        # Body ghosts at EVERY timestep (continuous chain of ellipses + arms).
        for t in range(T):
            a = 0.30 + 0.45 * (t / max(T - 1, 1))
            _draw_body_ghost(
                ax,
                float(states[t, 0]), float(states[t, 1]),
                float(states[t, 2]), float(states[t, 3]),
                float(states[t, 4]),
                float(states[t, 5]), float(states[t, 6]),
                alpha=a,
                zorder=6 + 0.01 * t,
            )

        # Mark collision points.
        for t in range(T):
            if env.check_collision(states[t]):
                ax.plot(states[t, 0], states[t, 1], "x",
                        color="red", markersize=8, zorder=20)

        # Height colorbar (matches ellipse fill color).
        import matplotlib.cm as cm
        from genedynamics.envs.humanoid_corridor_2d import H_MIN, H_MAX
        sm = plt.cm.ScalarMappable(cmap="coolwarm_r", norm=plt.Normalize(vmin=H_MIN, vmax=H_MAX))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=1.0, pad=0.02, aspect=30)
        cbar.set_label("Body height (m)", fontsize=7)

        # Legend for obstacle types.
        legend_patches = [
            mpatches.Patch(color=_Z_COLOR["full"][0], alpha=_Z_COLOR["full"][1], label="Full-height"),
            mpatches.Patch(color=_Z_COLOR["low"][0], alpha=_Z_COLOR["low"][1], label="Low bar (duck)"),
            mpatches.Patch(color=_Z_COLOR["floor"][0], alpha=_Z_COLOR["floor"][1], label="Floor block"),
            mpatches.Patch(color=_Z_COLOR["mid"][0], alpha=_Z_COLOR["mid"][1], label="Aerial (sidestep)"),
        ]
        ax.legend(handles=legend_patches, loc="upper left", fontsize=6, framealpha=0.7)
        ax.set_xlabel("x (m)", fontsize=8)
        ax.set_ylabel("y (m)", fontsize=8)
        ax.set_title("Corridor Trajectory (top-down)", fontsize=10)
        ax.tick_params(labelsize=7)

    def save(self, output_path: Any, fig: Any, **kwargs: Any) -> None:
        dpi = kwargs.get("dpi", 150)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
