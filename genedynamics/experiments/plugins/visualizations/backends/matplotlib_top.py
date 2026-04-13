"""
matplotlib top-down backend.

Byte-compatible port of the original draw code from
``corridor_trajectory.py`` and ``stepping_stones_trajectory.py``.

Both render functions consume only ``SceneIR`` + ``list[RobotPoseIR]``
plus an ``extras`` dict for env-specific knobs (collision mask, foot
radius, gif state, ...). Future backends (iso, pyrender) live next to
this one and accept the same inputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge

from ..scene_ir import RobotPoseIR, SceneIR


# ---------------------------------------------------------------------------
# Corridor: shared style
# ---------------------------------------------------------------------------
_Z_COLOR = {
    "full":  ("#4a5568", 0.85),
    "low":   ("#e53e3e", 0.55),
    "floor": ("#dd6b20", 0.55),
    "mid":   ("#805ad5", 0.55),
}


def _obs_type(z_lo: float, z_hi: float) -> str:
    if z_lo <= 0.05 and z_hi >= 1.8:
        return "full"
    if z_lo > 0.4:
        return "low"
    if z_hi < 0.5:
        return "floor"
    return "mid"


def _draw_corridor_scene(ax: plt.Axes, scene: SceneIR) -> None:
    xmin, xmax, ymin, ymax = scene.bounds
    length = xmax - xmin
    hw = (ymax - ymin) / 2.0

    ax.set_facecolor("#f7f9fc")
    ax.set_aspect("equal")
    ax.set_xlim(xmin - 0.3, xmax + 0.3)
    ax.set_ylim(ymin - 0.3, ymax + 0.3)

    # Floor
    ax.add_patch(plt.Rectangle(
        (xmin, ymin), length, 2 * hw,
        facecolor="#e2e8f0", edgecolor="none", zorder=0,
    ))

    # Walls
    wall_w = 0.08
    for wy in (ymin - wall_w, ymax):
        ax.add_patch(plt.Rectangle(
            (xmin, wy), length, wall_w,
            facecolor="#2d3748", edgecolor="#1a202c", linewidth=0.8, zorder=5,
        ))

    # Build a quick lookup so qc/sphere obstacles aren't double-drawn as boxes.
    qc_keys = {(round(qc.cx, 4), round(qc.cy, 4), round(qc.radius, 4))
               for qc in scene.quarter_circles}
    sph_keys = {(round(s.cx, 4), round(s.cy, 4), round(s.radius, 4))
                for s in scene.spheres}

    # Spheres
    for s in scene.spheres:
        otype = _obs_type(s.z_min, s.z_max)
        fc, alpha = _Z_COLOR[otype]
        ax.add_patch(plt.Circle(
            (s.cx, s.cy), s.radius,
            facecolor=fc, edgecolor="#1a202c", linewidth=0.6,
            alpha=alpha, zorder=3,
        ))

    # Quarter circles
    for qc in scene.quarter_circles:
        otype = _obs_type(qc.z_min, qc.z_max)
        fc, alpha = _Z_COLOR[otype]
        cs = qc.clip_sign
        if qc.cy > 0:
            theta1, theta2 = (180, 270) if cs > 0 else (270, 360)
        else:
            theta1, theta2 = (90, 180) if cs > 0 else (0, 90)
        ax.add_patch(Wedge(
            (qc.cx, qc.cy), qc.radius, theta1, theta2,
            facecolor=fc, edgecolor="#1a202c", linewidth=0.6,
            alpha=alpha, zorder=3,
        ))

    # Boxes (skip ones whose AABB matches a sphere/qc we already drew)
    for b in scene.boxes:
        cx_b = 0.5 * (b.x_min + b.x_max)
        cy_b = 0.5 * (b.y_min + b.y_max)
        r_b = 0.5 * (b.x_max - b.x_min)
        key = (round(cx_b, 4), round(cy_b, 4), round(r_b, 4))
        if key in sph_keys or key in qc_keys:
            # also draw the box-style label fallback below
            pass
        else:
            otype = _obs_type(b.z_min, b.z_max)
            fc, alpha = _Z_COLOR[otype]
            w = b.x_max - b.x_min
            h = b.y_max - b.y_min
            ax.add_patch(plt.Rectangle(
                (b.x_min, b.y_min), w, h,
                facecolor=fc, edgecolor="#1a202c", linewidth=0.6,
                alpha=alpha, zorder=3,
            ))
        # Labels (matching original behavior: uses AABB centroid for all shapes)
        w_obs = b.x_max - b.x_min
        h_obs = b.y_max - b.y_min
        if w_obs > 0.15 and h_obs > 0.06:
            ax.text(cx_b, cy_b, b.name, fontsize=4, ha="center", va="center",
                    color="white", zorder=4, clip_on=True)

    # Start / Goal markers
    if scene.start is not None:
        ax.plot(scene.start[0], scene.start[1], "o", color="#38a169",
                markersize=10, zorder=10, label="start")
    if scene.goal is not None:
        ax.plot(scene.goal[0], scene.goal[1], "*", color="#e53e3e",
                markersize=14, zorder=10, label="goal")


def _draw_body_ghost(
    ax: plt.Axes,
    pose: RobotPoseIR,
    *,
    torso_a: float,
    torso_b: float,
    torso_crouch_extra: float,
    h_nominal: float,
    h_min: float,
    h_max: float,
    arm_reach_open: float,
    arm_reach_tucked: float,
    alpha: float = 0.4,
    zorder: float = 8,
) -> None:
    heading = pose.yaw + pose.torso_yaw
    a_eff = torso_a + torso_crouch_extra * max(0.0, h_nominal - pose.body_height)
    h_norm = np.clip((pose.body_height - h_min) / max(h_max - h_min, 1e-6), 0.0, 1.0)
    h_color = cm.coolwarm_r(h_norm)
    ellipse = mpatches.Ellipse(
        (pose.x, pose.y), width=2 * torso_b, height=2 * a_eff,
        angle=np.degrees(heading),
        facecolor=h_color, edgecolor="#2c5282",
        alpha=alpha, linewidth=0.6, zorder=zorder,
    )
    ax.add_patch(ellipse)

    c, s = np.cos(heading), np.sin(heading)
    for sign, a_tuck, color in [(1.0, pose.arm_tuck_L, "#e53e3e"),
                                 (-1.0, pose.arm_tuck_R, "#dd6b20")]:
        reach = arm_reach_open + (arm_reach_tucked - arm_reach_open) * np.clip(a_tuck, 0, 1)
        display_reach = max(reach, 0.12)
        lx, ly = 0.0, sign * display_reach
        wx = pose.x + c * lx - s * ly
        wy = pose.y + s * lx + c * ly
        lw = 2.0 if a_tuck > 0.5 else 1.2
        ax.plot([pose.x, wx], [pose.y, wy], color=color, linewidth=lw,
                alpha=alpha * 0.9, zorder=zorder + 0.1)
        ms = 4.0 if a_tuck > 0.5 else 2.5
        ax.plot(wx, wy, "o", color=color, markersize=ms,
                alpha=alpha * 0.9, zorder=zorder + 0.2)


def render_corridor_top(
    fig: Any,
    ax: plt.Axes,
    scene: SceneIR,
    poses: List[RobotPoseIR],
    extras: Optional[Dict[str, Any]] = None,
) -> None:
    """Render corridor scene + body ghosts in the original top-down style.

    Required ``extras`` keys:
      - torso_a, torso_b, torso_crouch_extra
      - h_nominal, h_min, h_max
      - arm_reach_open, arm_reach_tucked
      - collision_mask (bool list/array, length T) — optional
    """
    extras = extras or {}
    _draw_corridor_scene(ax, scene)

    if not poses:
        return

    xs = np.array([p.x for p in poses], dtype=np.float32)
    ys = np.array([p.y for p in poses], dtype=np.float32)
    T = xs.shape[0]

    # Trajectory line
    ax.plot(xs, ys, "-", color="#3182ce", linewidth=1.8,
            alpha=0.5, zorder=6, label="trajectory")

    # Per-frame ghosts
    for t, pose in enumerate(poses):
        a = 0.30 + 0.45 * (t / max(T - 1, 1))
        _draw_body_ghost(
            ax, pose,
            torso_a=extras["torso_a"], torso_b=extras["torso_b"],
            torso_crouch_extra=extras["torso_crouch_extra"],
            h_nominal=extras["h_nominal"],
            h_min=extras["h_min"], h_max=extras["h_max"],
            arm_reach_open=extras["arm_reach_open"],
            arm_reach_tucked=extras["arm_reach_tucked"],
            alpha=a, zorder=6 + 0.01 * t,
        )

    # Collision marks
    coll = extras.get("collision_mask")
    if coll is not None:
        coll = np.asarray(coll).astype(bool).reshape(-1)
        for t in range(min(T, coll.shape[0])):
            if coll[t]:
                ax.plot(xs[t], ys[t], "x", color="red",
                        markersize=8, zorder=20)

    # Height colorbar
    sm = plt.cm.ScalarMappable(
        cmap="coolwarm_r",
        norm=plt.Normalize(vmin=extras["h_min"], vmax=extras["h_max"]),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=1.0, pad=0.02, aspect=30)
    cbar.set_label("Body height (m)", fontsize=7)

    # Legend
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


# ---------------------------------------------------------------------------
# Stepping stones
# ---------------------------------------------------------------------------
def draw_stepping_scene(ax: plt.Axes, scene: SceneIR) -> None:
    """Public helper: draw a stepping-stones scene background only."""
    _draw_stepping_scene(ax, scene)


def draw_corridor_scene(ax: plt.Axes, scene: SceneIR) -> None:
    """Public helper: draw a corridor scene background only."""
    _draw_corridor_scene(ax, scene)


def _draw_stepping_scene(ax: plt.Axes, scene: SceneIR) -> None:
    xmin, xmax, ymin, ymax = scene.bounds
    ax.set_aspect("equal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_facecolor("#f7f9fc")
    if scene.river is not None:
        rx0, rx1, ry0, ry1 = scene.river
        ax.add_patch(plt.Rectangle(
            (rx0, ry0), rx1 - rx0, ry1 - ry0,
            facecolor="#d9ecff", edgecolor="#7aa6d9",
            linewidth=1.0, alpha=0.9, zorder=0.5,
        ))
    for plat in scene.platforms:
        ax.add_patch(plt.Rectangle(
            (plat.x_min, plat.y_min),
            plat.x_max - plat.x_min, plat.y_max - plat.y_min,
            facecolor="#8b949e", edgecolor="#2f3640",
            linewidth=1.4, alpha=0.88, zorder=0.85,
        ))
    for st in scene.stones:
        ax.add_patch(plt.Circle(
            (st.cx, st.cy), st.radius,
            facecolor="#9099a2", edgecolor="#59636e",
            linewidth=0.7, alpha=0.8, zorder=1,
        ))


def render_stepping_top(
    fig: Any,
    ax: plt.Axes,
    scene: SceneIR,
    poses: List[RobotPoseIR],
    extras: Optional[Dict[str, Any]] = None,
) -> None:
    """Render stepping-stones scene + body/feet trajectories.

    Required ``extras`` keys:
      - foot_radius (float)
      - stone_margin (float)
      - violation_fn (callable: (p, centers, radii, platforms, margin) -> float)
      - centers (np.ndarray (N,2)), radii (np.ndarray (N,))
      - platforms (np.ndarray (M,4))
    Optional:
      - partial_until_step (int | None)
      - gif_style (bool)
    """
    extras = extras or {}
    _draw_stepping_scene(ax, scene)

    if not poses:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.25)
        return

    body = np.array([(p.x, p.y) for p in poses], dtype=np.float32)
    leg_order = ("FL", "FR", "RL", "RR")
    leg_colors = {"FL": "#ff7f0e", "FR": "#2ca02c",
                  "RL": "#9467bd", "RR": "#8c564b"}
    feet_full = {leg: np.array(
        [(p.feet[leg][0], p.feet[leg][1]) if (p.feet and leg in p.feet)
         else (np.nan, np.nan) for p in poses], dtype=np.float32)
        for leg in leg_order}

    T_all = body.shape[0]
    partial_t = extras.get("partial_until_step")
    if partial_t is not None:
        T_vis = min(int(partial_t) + 1, T_all)
    else:
        T_vis = T_all
    gif_style = bool(extras.get("gif_style", False))

    foot_radius = float(extras.get("foot_radius", 0.03))
    centers_v = np.asarray(extras.get("centers", np.zeros((0, 2))), dtype=np.float32)
    radii_v = np.asarray(extras.get("radii", np.zeros((0,))), dtype=np.float32)
    plat_v = np.asarray(extras.get("platforms", np.zeros((0, 4))), dtype=np.float32)
    sm_v = float(extras.get("stone_margin", 0.032))
    violation_fn = extras.get("violation_fn")

    def _on_platform_raw(p):
        for k in range(plat_v.shape[0]):
            xmin, xmax, ymin, ymax = (float(plat_v[k, 0]), float(plat_v[k, 1]),
                                       float(plat_v[k, 2]), float(plat_v[k, 3]))
            if xmin <= float(p[0]) <= xmax and ymin <= float(p[1]) <= ymax:
                return True
        return False

    def _on_any_stone_raw(p):
        if centers_v.shape[0] == 0:
            return False
        d = np.linalg.norm(centers_v - p[None, :], axis=-1)
        return bool(np.any(d <= radii_v))

    if gif_style and partial_t is not None:
        ax.plot(body[:, 0], body[:, 1], color="#1664c0",
                linewidth=1.0, alpha=0.20, zorder=2.5)
        for leg in leg_order:
            pts_full = feet_full[leg]
            ax.plot(pts_full[:, 0], pts_full[:, 1], color=leg_colors[leg],
                    linewidth=0.6, alpha=0.12, zorder=2.3)

    ax.plot(body[:T_vis, 0], body[:T_vis, 1], color="#1664c0",
            linewidth=1.8, alpha=0.9, zorder=3, label="body")

    for leg in leg_order:
        pts_full = feet_full[leg]
        pts = pts_full[:T_vis]
        ax.plot(pts[:, 0], pts[:, 1], color=leg_colors[leg],
                linewidth=1.2, alpha=0.80, zorder=2.7, label=leg)
        ax.scatter(pts[:, 0], pts[:, 1], s=8,
                   color=leg_colors[leg], alpha=0.18, zorder=2.5)
        if violation_fn is not None:
            for ti in range(int(pts.shape[0])):
                pt = pts[ti]
                if np.any(np.isnan(pt)):
                    continue
                if plat_v.shape[0] > 0 and _on_platform_raw(pt):
                    continue
                if _on_any_stone_raw(pt):
                    continue
                if violation_fn(pt, centers_v, radii_v, plat_v, sm_v) > 1e-3:
                    ax.scatter(
                        float(pt[0]), float(pt[1]),
                        s=42, facecolors="#d62728", edgecolors="#6a0000",
                        linewidths=0.65, marker="o", zorder=4.2,
                    )

    # Swing-pair halos
    for t in range(T_vis):
        pose = poses[t]
        for swing_leg in pose.swing_legs:
            if pose.feet is None or swing_leg not in pose.feet:
                continue
            p = pose.feet[swing_leg]
            ax.add_patch(plt.Circle(
                (float(p[0]), float(p[1])),
                float(1.5 * foot_radius),
                facecolor="#f04f88", edgecolor="none",
                alpha=0.08, zorder=2.1,
            ))

    ax.scatter(body[0, 0], body[0, 1], marker="o", s=28,
               color="white", edgecolor="#1664c0", zorder=5)
    if gif_style and partial_t is not None and T_vis > 0:
        t_cur = T_vis - 1
        ax.scatter(body[t_cur, 0], body[t_cur, 1], marker="o", s=50,
                   color="#1664c0", edgecolor="white",
                   linewidths=1.0, zorder=6)
        for leg in leg_order:
            pts_full = feet_full[leg]
            ax.scatter(pts_full[t_cur, 0], pts_full[t_cur, 1], s=30,
                       color=leg_colors[leg], edgecolor="white",
                       linewidths=0.5, zorder=5.5)
    else:
        ax.scatter(body[-1, 0], body[-1, 1], marker="*",
                   s=90, color="#f04f88", zorder=5)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.25)
