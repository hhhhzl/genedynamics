"""
matplotlib isometric (Tier 2 pseudo-3D) backend.

Uses a fixed isometric projection (viewer at +x+y+z, up = +z) and a
back-to-front painter sort via matplotlib zorder. No new dependencies
beyond matplotlib + numpy.

Coordinate convention
---------------------
World axes: x forward (along corridor / along map), y lateral, z up.
Projection:
    u = (x - y) * cos30
    v = -(x + y) * sin30 + z
Larger (x + y + z) ⇒ closer to viewer ⇒ higher zorder.

What gets drawn
---------------
Corridor:
  * floor polygon + iso grid
  * back wall (y = -hw) only — the +y wall is skipped so the room
    interior is visible (architectural cutaway)
  * box / sphere / qc obstacles, all 3 visible faces of each box
  * start / goal disks
  * trajectory polyline at z ≈ 0.01
  * G1 ghosts: one oriented torso prism + head + two arm segments per
    timestep, height = body_height (crouch visible), yaw = yaw +
    torso_yaw (sidestep visible), arm length driven by arm_tuck

Stepping stones:
  * floor + river polygon + grid
  * support platforms as low extruded boxes
  * stones as thin disks (top ellipse + slight side band)
  * body trajectory + per-leg foot tracks
  * a simple G1 ghost at the last frame (no torso_yaw/arms data, so
    drawn as a plain forward-facing prism)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

from ..scene_ir import RobotPoseIR, SceneIR


# ---------------------------------------------------------------------------
# Camera (mutable per-render). Default is standard 30/30 isometric. Mutate
# via _set_camera() at the top of each render entry point; matplotlib is
# single-threaded so module-level state is fine here.
# ---------------------------------------------------------------------------
_CAM = {
    "yaw": np.deg2rad(30.0),    # rotation around world z (azimuth)
    "pitch": np.deg2rad(30.0),  # tilt below horizon
}


def _set_camera(yaw_deg: float, pitch_deg: float) -> None:
    _CAM["yaw"] = float(np.deg2rad(yaw_deg))
    _CAM["pitch"] = float(np.deg2rad(pitch_deg))


def _viewer_dir() -> np.ndarray:
    yaw = _CAM["yaw"]
    pitch = _CAM["pitch"]
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    # Camera looks toward origin from (sin(yaw)*cos(pitch), cos(yaw)*cos(pitch), sin(pitch))
    return np.array([sy * cp, cy * cp, sp], dtype=np.float32)


def iso(x: float, y: float, z: float = 0.0) -> tuple:
    yaw = _CAM["yaw"]; pitch = _CAM["pitch"]
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    u = x * cy - y * sy
    v = -(x * sy + y * cy) * sp + z * cp
    return (float(u), float(v))


def iso_arr(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    yaw = _CAM["yaw"]; pitch = _CAM["pitch"]
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    u = pts[:, 0] * cy - pts[:, 1] * sy
    v = -(pts[:, 0] * sy + pts[:, 1] * cy) * sp + pts[:, 2] * cp
    return np.column_stack([u, v])


# Backward-compat constants used by sphere/stone width-stretch heuristics.
COS30 = float(np.cos(np.pi / 6))
SIN30 = float(np.sin(np.pi / 6))


def _depth_zorder(x: float, y: float, z: float, base: float = 10.0) -> float:
    """Larger projection along the camera direction → closer → higher zorder."""
    v = _viewer_dir()
    return base + (x * v[0] + y * v[1] + z * v[2]) * 10.0


def _shade(color, factor: float):
    rgb = mcolors.to_rgb(color)
    return tuple(max(0.0, min(1.0, c * factor)) for c in rgb)


# ---------------------------------------------------------------------------
# Box / face primitives
# ---------------------------------------------------------------------------
def _draw_box(
    ax: plt.Axes,
    xmin: float, xmax: float,
    ymin: float, ymax: float,
    zmin: float, zmax: float,
    *,
    top_color,
    side_color=None,
    edge: str = "#1a202c",
    alpha: float = 1.0,
    base_zorder: float = 10.0,
    edge_lw: float = 0.5,
) -> None:
    """Draw the 3 viewer-facing faces (top, +x, +y) of an axis-aligned box."""
    side_color = side_color if side_color is not None else top_color
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)
    z_order = _depth_zorder(cx, cy, cz, base=base_zorder)

    # +z (top)
    top = np.array([
        [xmin, ymin, zmax], [xmax, ymin, zmax],
        [xmax, ymax, zmax], [xmin, ymax, zmax],
    ], dtype=np.float32)
    # +x (right)
    px = np.array([
        [xmax, ymin, zmin], [xmax, ymax, zmin],
        [xmax, ymax, zmax], [xmax, ymin, zmax],
    ], dtype=np.float32)
    # +y (back)
    py = np.array([
        [xmin, ymax, zmin], [xmax, ymax, zmin],
        [xmax, ymax, zmax], [xmin, ymax, zmax],
    ], dtype=np.float32)

    for face3d, shade_factor in [
        (top, 1.00),     # top brightest
        (px,  0.78),     # +x face medium
        (py,  0.62),     # +y face darkest
    ]:
        fc = top_color if shade_factor >= 0.99 else _shade(side_color, shade_factor)
        face2d = iso_arr(face3d)
        ax.add_patch(Polygon(
            face2d, facecolor=fc, edgecolor=edge,
            linewidth=edge_lw, alpha=alpha, zorder=z_order,
        ))


def _draw_floor(
    ax: plt.Axes,
    xmin: float, xmax: float, ymin: float, ymax: float,
    *,
    color: str = "#e8eef5",
    grid_color: str = "#cbd5e0",
    grid_step: float = 0.5,
    draw_grid: bool = True,
) -> None:
    corners = np.array([
        [xmin, ymin, 0.0], [xmax, ymin, 0.0],
        [xmax, ymax, 0.0], [xmin, ymax, 0.0],
    ], dtype=np.float32)
    p2d = iso_arr(corners)
    ax.add_patch(Polygon(
        p2d, facecolor=color, edgecolor="#a0aec0",
        linewidth=0.4, zorder=0,
    ))
    if not draw_grid:
        return
    xs = np.arange(xmin, xmax + 1e-6, grid_step)
    ys = np.arange(ymin, ymax + 1e-6, grid_step)
    for x in xs:
        a = iso(float(x), float(ymin), 0.0)
        b = iso(float(x), float(ymax), 0.0)
        ax.plot([a[0], b[0]], [a[1], b[1]],
                color=grid_color, linewidth=0.3, zorder=0.1)
    for y in ys:
        a = iso(float(xmin), float(y), 0.0)
        b = iso(float(xmax), float(y), 0.0)
        ax.plot([a[0], b[0]], [a[1], b[1]],
                color=grid_color, linewidth=0.3, zorder=0.1)


def _setup_iso_axes(
    ax: plt.Axes,
    xmin: float, xmax: float, ymin: float, ymax: float,
    z_max: float = 2.0,
    pad: float = 0.4,
) -> None:
    corners = np.array([
        [xmin, ymin, 0.0], [xmax, ymin, 0.0],
        [xmin, ymax, 0.0], [xmax, ymax, 0.0],
        [xmin, ymin, z_max], [xmax, ymin, z_max],
        [xmin, ymax, z_max], [xmax, ymax, z_max],
    ], dtype=np.float32)
    p2d = iso_arr(corners)
    ax.set_xlim(p2d[:, 0].min() - pad, p2d[:, 0].max() + pad)
    ax.set_ylim(p2d[:, 1].min() - pad, p2d[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f7f9fc")


# ---------------------------------------------------------------------------
# Corridor obstacle styling (mirror top backend palette)
# ---------------------------------------------------------------------------
_Z_COLOR = {
    "full":  ("#4a5568", 0.55),
    "low":   ("#e53e3e", 0.45),
    "floor": ("#dd6b20", 0.55),
    "mid":   ("#805ad5", 0.50),
}


def _obs_type(z_lo: float, z_hi: float) -> str:
    if z_lo <= 0.05 and z_hi >= 1.8:
        return "full"
    if z_lo > 0.4:
        return "low"
    if z_hi < 0.5:
        return "floor"
    return "mid"


# ---------------------------------------------------------------------------
# G1 ghost
# ---------------------------------------------------------------------------
def _draw_g1_iso(
    ax: plt.Axes,
    pose: RobotPoseIR,
    *,
    torso_a: float,
    torso_b: float,
    h_min: float,
    h_max: float,
    arm_reach_open: float,
    arm_reach_tucked: float,
    alpha: float = 0.95,
    zorder_boost: float = 1000.0,
    anchor_line: bool = True,
) -> None:
    """Draw G1 as oriented torso prism + head + arms (+ leg stubs)."""
    x, y = float(pose.x), float(pose.y)
    h = float(pose.body_height)
    heading = float(pose.yaw + pose.torso_yaw)

    a, b = torso_a, torso_b

    # Torso 8 corners in body frame (origin at floor center under torso)
    local_corners = np.array([
        [-a, -b, 0.0], [ a, -b, 0.0], [ a,  b, 0.0], [-a,  b, 0.0],
        [-a, -b,   h], [ a, -b,   h], [ a,  b,   h], [-a,  b,   h],
    ], dtype=np.float32)
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    world = local_corners @ R.T + np.array([x, y, 0.0], dtype=np.float32)

    # Color by height
    h_norm = float(np.clip((h - h_min) / max(h_max - h_min, 1e-6), 0.0, 1.0))
    body_color = cm.coolwarm_r(h_norm)

    # Per-face: indices (in `world`), body-frame normal, shade factor
    faces = [
        ([4, 5, 6, 7], np.array([0.0, 0.0,  1.0]), 1.00),  # top
        ([0, 3, 2, 1], np.array([0.0, 0.0, -1.0]), 0.55),  # bottom (rarely visible)
        ([0, 1, 5, 4], np.array([0.0, -1.0, 0.0]), 0.78),  # -y body
        ([3, 7, 6, 2], np.array([0.0,  1.0, 0.0]), 0.78),  # +y body
        ([0, 4, 7, 3], np.array([-1.0, 0.0, 0.0]), 0.65),  # -x body
        ([1, 2, 6, 5], np.array([ 1.0, 0.0, 0.0]), 0.65),  # +x body
    ]
    z_base = _depth_zorder(x, y, h * 0.5, base=zorder_boost)

    # Anchor line: faint vertical from ground to torso top, makes ghosts
    # readable even when overlapping translucent obstacles.
    if anchor_line:
        a0 = iso(x, y, 0.0)
        a1 = iso(x, y, h + 0.12)
        ax.plot([a0[0], a1[0]], [a0[1], a1[1]],
                color="#1a202c", linewidth=0.6,
                alpha=alpha * 0.5, zorder=z_base - 0.5)

    vdir = _viewer_dir()
    for idx, n_body, shade_factor in faces:
        n_world = R @ n_body
        if float(np.dot(n_world, vdir)) <= 1e-6:
            continue
        face2d = iso_arr(world[idx])
        fc = body_color if shade_factor >= 0.99 else _shade(body_color, shade_factor)
        ax.add_patch(Polygon(
            face2d, facecolor=fc, edgecolor="#2c5282",
            linewidth=0.7, alpha=alpha, zorder=z_base,
        ))

    # Head
    head_r = 0.10
    hp = iso(x, y, h + head_r)
    ax.add_patch(Circle(
        (hp[0], hp[1]), head_r * COS30 * 1.2,
        facecolor="#fbbf24", edgecolor="#92400e",
        linewidth=0.6, alpha=alpha, zorder=z_base + 1.0,
    ))

    # Heading arrow on torso top: shows yaw + torso_yaw direction.
    # Drawn from torso center forward by `torso_a` along body +x (forward).
    arrow_len = torso_a * 1.1
    tip_body = np.array([arrow_len, 0.0, h + 0.01], dtype=np.float32)
    tail_body = np.array([-arrow_len * 0.2, 0.0, h + 0.01], dtype=np.float32)
    tip_world = R @ tip_body + np.array([x, y, 0.0], dtype=np.float32)
    tail_world = R @ tail_body + np.array([x, y, 0.0], dtype=np.float32)
    tail_2d = iso_arr(tail_world[None])[0]
    tip_2d = iso_arr(tip_world[None])[0]
    ax.annotate(
        "", xy=(tip_2d[0], tip_2d[1]), xytext=(tail_2d[0], tail_2d[1]),
        arrowprops=dict(
            arrowstyle="->,head_length=0.35,head_width=0.25",
            color="#1a202c", lw=1.0, alpha=alpha,
            shrinkA=0.0, shrinkB=0.0,
        ),
        zorder=z_base + 2.0,
    )

    # Arms
    shoulder_z = h * 0.88
    for sign, a_tuck, color in [
        (1.0,  float(pose.arm_tuck_L), "#e53e3e"),
        (-1.0, float(pose.arm_tuck_R), "#dd6b20"),
    ]:
        reach = arm_reach_open + (arm_reach_tucked - arm_reach_open) * float(np.clip(a_tuck, 0.0, 1.0))
        sh_body = np.array([0.0, sign * b, shoulder_z], dtype=np.float32)
        hd_body = np.array([0.0, sign * (b + max(reach, 0.05)), shoulder_z], dtype=np.float32)
        sh_world = R @ sh_body + np.array([x, y, 0.0], dtype=np.float32)
        hd_world = R @ hd_body + np.array([x, y, 0.0], dtype=np.float32)
        sh2d = iso_arr(sh_world[None])[0]
        hd2d = iso_arr(hd_world[None])[0]
        lw = 3.2 if a_tuck > 0.5 else 2.2
        ax.plot([sh2d[0], hd2d[0]], [sh2d[1], hd2d[1]],
                color=color, linewidth=lw, alpha=alpha,
                solid_capstyle="round", zorder=z_base + 1.5)
        ax.plot(hd2d[0], hd2d[1], "o", color=color,
                markersize=4.5 if a_tuck > 0.5 else 3.0,
                alpha=alpha, zorder=z_base + 1.6)

    # Leg stubs (from torso bottom corners to ground)
    for sign in (-1.0, 1.0):
        hip_body = np.array([0.0, sign * b * 0.5, 0.0], dtype=np.float32)
        hip_world = R @ hip_body + np.array([x, y, 0.0], dtype=np.float32)
        hp2d = iso_arr(hip_world[None])[0]
        # Foot directly below hip
        fp_world = hip_world.copy()
        fp_world[2] = 0.0
        fp2d = iso_arr(fp_world[None])[0]
        # Already at z=0; draw a small ground marker
        ax.plot(fp2d[0], fp2d[1], "o", color="#2c5282",
                markersize=3.0, alpha=alpha * 0.9,
                zorder=z_base + 0.5)


# ---------------------------------------------------------------------------
# Corridor renderer
# ---------------------------------------------------------------------------
def render_corridor_iso(
    fig: Any,
    ax: plt.Axes,
    scene: SceneIR,
    poses: List[RobotPoseIR],
    extras: Optional[Dict[str, Any]] = None,
) -> None:
    extras = extras or {}
    _set_camera(
        yaw_deg=float(extras.get("camera_yaw_deg", 35.0)),
        pitch_deg=float(extras.get("camera_pitch_deg", 28.0)),
    )
    xmin, xmax, ymin, ymax = scene.bounds

    # Optional auto figure resize. Corridor is long along x; aim for ~2:1.
    if extras.get("auto_resize_fig", True):
        try:
            length = xmax - xmin
            width = max(ymax - ymin, 1.0)
            target_w = 11.0
            target_h = max(5.5, target_w * (1.6 * width + 1.0) / max(length, 1.0))
            fig.set_size_inches(target_w, target_h, forward=True)
        except Exception:
            pass

    # Auto z_room from obstacle z_max (cap at 2.0, floor at 1.0).
    z_obs = [b.z_max for b in scene.boxes if b.shape == "box"]
    z_obs += [s.z_max for s in scene.spheres]
    z_obs += [qc.z_max for qc in scene.quarter_circles]
    z_room = float(min(2.0, max(1.0, max(z_obs) * 1.05))) if z_obs else 1.6
    wall_h = z_room - 0.05

    _setup_iso_axes(ax, xmin, xmax, ymin, ymax, z_max=z_room)
    _draw_floor(ax, xmin, xmax, ymin, ymax, draw_grid=False)

    # Back wall = the y = ymax wall (the far side from the viewer, who is
    # at +x+y+z). The viewer-side wall (y=ymin) is intentionally omitted as
    # an architectural cutaway so the corridor interior is visible.
    wall_thick = 0.08
    _draw_box(
        ax,
        xmin, xmax,
        ymax, ymax + wall_thick,
        0.0, wall_h,
        top_color="#4a5568", side_color="#4a5568",
        edge="#1a202c", alpha=0.85,
        base_zorder=5.0,
    )

    # Box obstacles (skip qc/sphere proxies via shape tag)
    for b in scene.boxes:
        if b.shape != "box":
            continue
        otype = _obs_type(b.z_min, b.z_max)
        fc, alpha = _Z_COLOR[otype]
        _draw_box(
            ax,
            b.x_min, b.x_max, b.y_min, b.y_max,
            b.z_min, b.z_max,
            top_color=fc, side_color=fc,
            alpha=alpha, base_zorder=20.0,
        )

    # Sphere obstacles: ground shadow disk + tethered ball + filled circle.
    for s in scene.spheres:
        otype = _obs_type(s.z_min, s.z_max)
        fc, alpha = _Z_COLOR[otype]
        cz = 0.5 * (s.z_min + s.z_max)

        # Ground shadow (squashed ellipse on the floor)
        sh_u, sh_v = iso(s.cx, s.cy, 0.0)
        ax.add_patch(mpatches.Ellipse(
            (sh_u, sh_v),
            width=2 * s.radius * COS30 * 1.4,
            height=s.radius * 0.7,
            facecolor="#1a202c", edgecolor="none",
            alpha=0.18,
            zorder=_depth_zorder(s.cx, s.cy, 0.0, base=15.0),
        ))

        # Tether line from ground shadow up to sphere center (shows altitude)
        cu, cv = iso(s.cx, s.cy, cz)
        ax.plot([sh_u, cu], [sh_v, cv],
                color="#4a5568", linewidth=0.6,
                linestyle=":", alpha=0.6,
                zorder=_depth_zorder(s.cx, s.cy, cz * 0.5, base=24.0))

        # Sphere as a filled circle in screen space (radius preserved)
        ax.add_patch(Circle(
            (cu, cv), s.radius,
            facecolor=fc, edgecolor="#1a202c",
            linewidth=0.7, alpha=min(1.0, alpha + 0.20),
            zorder=_depth_zorder(s.cx, s.cy, cz, base=25.0),
        ))
        # Highlight crescent for shading
        hl_u = cu - s.radius * 0.30
        hl_v = cv + s.radius * 0.30
        ax.add_patch(Circle(
            (hl_u, hl_v), s.radius * 0.55,
            facecolor=_shade(fc, 1.35), edgecolor="none",
            alpha=min(1.0, alpha + 0.20) * 0.55,
            zorder=_depth_zorder(s.cx, s.cy, cz, base=25.0) + 0.5,
        ))

    # Quarter circles: render as proper wedge wall (top fan + curved side).
    for qc in scene.quarter_circles:
        otype = _obs_type(qc.z_min, qc.z_max)
        fc, alpha = _Z_COLOR[otype]

        # Determine the angle range matching the top backend's logic.
        if qc.cy > 0:
            theta1, theta2 = (np.pi, 1.5 * np.pi) if qc.clip_sign > 0 else (1.5 * np.pi, 2.0 * np.pi)
        else:
            theta1, theta2 = (0.5 * np.pi, np.pi) if qc.clip_sign > 0 else (0.0, 0.5 * np.pi)

        n_arc = 16
        thetas = np.linspace(theta1, theta2, n_arc)
        arc_x = qc.cx + qc.radius * np.cos(thetas)
        arc_y = qc.cy + qc.radius * np.sin(thetas)

        z_top = qc.z_max
        z_bot = qc.z_min

        # Top fan: center + arc points, projected
        top_pts3d = np.zeros((n_arc + 1, 3), dtype=np.float32)
        top_pts3d[0] = (qc.cx, qc.cy, z_top)
        top_pts3d[1:, 0] = arc_x
        top_pts3d[1:, 1] = arc_y
        top_pts3d[1:, 2] = z_top
        top_2d = iso_arr(top_pts3d)
        cx_b, cy_b = qc.cx, qc.cy
        z_order = _depth_zorder(cx_b, cy_b, 0.5 * (z_bot + z_top), base=20.0)
        ax.add_patch(Polygon(
            top_2d, facecolor=fc, edgecolor="#1a202c",
            linewidth=0.5, alpha=alpha, zorder=z_order + 0.3,
        ))

        # Curved side: triangle strip between arc_top and arc_bot
        for i in range(n_arc - 1):
            quad3d = np.array([
                [arc_x[i],     arc_y[i],     z_bot],
                [arc_x[i + 1], arc_y[i + 1], z_bot],
                [arc_x[i + 1], arc_y[i + 1], z_top],
                [arc_x[i],     arc_y[i],     z_top],
            ], dtype=np.float32)
            quad2d = iso_arr(quad3d)
            ax.add_patch(Polygon(
                quad2d, facecolor=_shade(fc, 0.72),
                edgecolor="none", alpha=alpha,
                zorder=_depth_zorder(arc_x[i], arc_y[i],
                                     0.5 * (z_bot + z_top), base=20.0),
            ))

    # Start / Goal
    if scene.start is not None:
        sp = iso(scene.start[0], scene.start[1], 0.02)
        ax.plot(sp[0], sp[1], "o", color="#38a169",
                markersize=10, zorder=50, label="start")
    if scene.goal is not None:
        gp = iso(scene.goal[0], scene.goal[1], 0.02)
        ax.plot(gp[0], gp[1], "*", color="#e53e3e",
                markersize=14, zorder=50, label="goal")

    # Trajectory line at floor level
    if poses:
        pts = np.array([[p.x, p.y, 0.02] for p in poses], dtype=np.float32)
        line2d = iso_arr(pts)
        ax.plot(line2d[:, 0], line2d[:, 1], "-",
                color="#3182ce", linewidth=1.6, alpha=0.55,
                zorder=8, label="trajectory")

    # G1 ghosts
    h_min = extras.get("h_min", 0.55)
    h_max = extras.get("h_max", 0.85)
    torso_a = extras.get("torso_a", 0.20)
    torso_b = extras.get("torso_b", 0.11)
    arm_open = extras.get("arm_reach_open", 0.35)
    arm_tucked = extras.get("arm_reach_tucked", 0.10)
    coll = extras.get("collision_mask")
    if coll is not None:
        coll = np.asarray(coll).astype(bool).reshape(-1)

    # Subsample for clarity if very long. Knobs are configurable via extras.
    n_ghosts = int(extras.get("n_ghosts", 12))
    ghost_alpha_lo = float(extras.get("ghost_alpha_min", 0.45))
    ghost_alpha_hi = float(extras.get("ghost_alpha_max", 0.95))
    T = len(poses)
    if T > 0 and n_ghosts > 0:
        step = max(1, T // n_ghosts)
        indices = list(range(0, T, step))
        if indices[-1] != T - 1:
            indices.append(T - 1)
        for k, t in enumerate(indices):
            ghost_alpha = (ghost_alpha_lo + (ghost_alpha_hi - ghost_alpha_lo)
                           * (k / max(len(indices) - 1, 1)))
            _draw_g1_iso(
                ax, poses[t],
                torso_a=torso_a, torso_b=torso_b,
                h_min=h_min, h_max=h_max,
                arm_reach_open=arm_open, arm_reach_tucked=arm_tucked,
                alpha=ghost_alpha,
                zorder_boost=1500.0 + k,
            )
            if coll is not None and t < coll.shape[0] and coll[t]:
                cp = iso(poses[t].x, poses[t].y, 0.05)
                ax.plot(cp[0], cp[1], "x", color="red",
                        markersize=10, zorder=400)

    # Height colorbar
    sm = plt.cm.ScalarMappable(
        cmap="coolwarm_r",
        norm=plt.Normalize(vmin=h_min, vmax=h_max),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02, aspect=30)
    cbar.set_label("Body height (m)", fontsize=7)

    legend_patches = [
        mpatches.Patch(color=_Z_COLOR["full"][0], alpha=_Z_COLOR["full"][1], label="Full-height"),
        mpatches.Patch(color=_Z_COLOR["low"][0], alpha=_Z_COLOR["low"][1], label="Low bar"),
        mpatches.Patch(color=_Z_COLOR["floor"][0], alpha=_Z_COLOR["floor"][1], label="Floor block"),
        mpatches.Patch(color=_Z_COLOR["mid"][0], alpha=_Z_COLOR["mid"][1], label="Aerial"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        fontsize=6,
        framealpha=0.7,
    )
    ax.set_title("Corridor Trajectory (isometric)", fontsize=10)


# ---------------------------------------------------------------------------
# Stepping stones renderer
# ---------------------------------------------------------------------------
def render_stepping_iso(
    fig: Any,
    ax: plt.Axes,
    scene: SceneIR,
    poses: List[RobotPoseIR],
    extras: Optional[Dict[str, Any]] = None,
) -> None:
    extras = extras or {}
    _set_camera(
        yaw_deg=float(extras.get("camera_yaw_deg", 35.0)),
        pitch_deg=float(extras.get("camera_pitch_deg", 30.0)),
    )
    xmin, xmax, ymin, ymax = scene.bounds
    _setup_iso_axes(ax, xmin, xmax, ymin, ymax, z_max=0.5)
    _draw_floor(ax, xmin, xmax, ymin, ymax, color="#eef2f7")

    # River as a slightly recessed flat polygon
    if scene.river is not None:
        rx0, rx1, ry0, ry1 = scene.river
        corners = np.array([
            [rx0, ry0, -0.02], [rx1, ry0, -0.02],
            [rx1, ry1, -0.02], [rx0, ry1, -0.02],
        ], dtype=np.float32)
        p2d = iso_arr(corners)
        ax.add_patch(Polygon(
            p2d, facecolor="#bfdcff", edgecolor="#7aa6d9",
            linewidth=1.0, alpha=0.92, zorder=0.4,
        ))

    # Platforms as low extruded boxes
    for p in scene.platforms:
        _draw_box(
            ax,
            p.x_min, p.x_max, p.y_min, p.y_max,
            0.0, 0.10,
            top_color="#9aa3ad", side_color="#8b949e",
            edge="#2f3640", alpha=0.92, edge_lw=0.9,
            base_zorder=15.0,
        )

    # Stones as thin disks: top ellipse + slim band underneath
    for st in scene.stones:
        h_top = 0.08
        cu_t, cv_t = iso(st.cx, st.cy, h_top)
        cu_b, cv_b = iso(st.cx, st.cy, 0.0)
        w = 2 * st.radius * COS30 * 1.6
        hgt = 2 * st.radius
        # Side band (darker) drawn first
        ax.add_patch(mpatches.Ellipse(
            (cu_b, cv_b), width=w, height=hgt,
            facecolor="#6e7681", edgecolor="#3a3f44",
            linewidth=0.5,
            zorder=_depth_zorder(st.cx, st.cy, 0.0, base=18.0),
        ))
        # Top
        ax.add_patch(mpatches.Ellipse(
            (cu_t, cv_t), width=w, height=hgt,
            facecolor="#9099a2", edgecolor="#59636e",
            linewidth=0.7,
            zorder=_depth_zorder(st.cx, st.cy, h_top, base=20.0),
        ))

    if not poses:
        return

    body = np.array([(p.x, p.y) for p in poses], dtype=np.float32)
    body3d = np.column_stack([body, np.full(body.shape[0], 0.12)])
    body2d = iso_arr(body3d)

    leg_order = ("FL", "FR", "RL", "RR")
    leg_colors = {"FL": "#ff7f0e", "FR": "#2ca02c",
                  "RL": "#9467bd", "RR": "#8c564b"}

    partial_t = extras.get("partial_until_step")
    T_all = body.shape[0]
    T_vis = min(int(partial_t) + 1, T_all) if partial_t is not None else T_all
    foot_radius = float(extras.get("foot_radius", 0.03))

    # Body line
    ax.plot(body2d[:T_vis, 0], body2d[:T_vis, 1],
            color="#1664c0", linewidth=1.8, alpha=0.9, zorder=60)

    # Per-leg foot traces
    for leg in leg_order:
        pts = np.array([
            (p.feet[leg][0], p.feet[leg][1], 0.10)
            if (p.feet and leg in p.feet) else (np.nan, np.nan, np.nan)
            for p in poses[:T_vis]
        ], dtype=np.float32)
        valid = ~np.isnan(pts[:, 0])
        if not np.any(valid):
            continue
        pts2d = iso_arr(pts[valid])
        ax.plot(pts2d[:, 0], pts2d[:, 1],
                color=leg_colors[leg], linewidth=1.2,
                alpha=0.85, zorder=55, label=leg)
        ax.scatter(pts2d[:, 0], pts2d[:, 1], s=10,
                   color=leg_colors[leg], alpha=0.5, zorder=56)

    # Swing-leg halos at each step
    for t in range(T_vis):
        for swing_leg in poses[t].swing_legs:
            if poses[t].feet is None or swing_leg not in poses[t].feet:
                continue
            fx, fy = poses[t].feet[swing_leg]
            up, vp = iso(fx, fy, 0.10)
            ax.add_patch(Circle(
                (up, vp), 1.5 * foot_radius * COS30 * 1.2,
                facecolor="#f04f88", edgecolor="none",
                alpha=0.18, zorder=54,
            ))

    # G1 ghost at last visible step (no torso_yaw / arm data, draw plain)
    if T_vis > 0:
        last = poses[T_vis - 1]
        ghost = RobotPoseIR(
            x=last.x, y=last.y, yaw=0.0, torso_yaw=0.0,
            body_height=0.78, arm_tuck_L=0.0, arm_tuck_R=0.0,
        )
        _draw_g1_iso(
            ax, ghost,
            torso_a=0.18, torso_b=0.10,
            h_min=0.55, h_max=0.85,
            arm_reach_open=0.30, arm_reach_tucked=0.10,
            alpha=0.85, zorder_boost=300.0,
        )

    ax.set_title("Stepping Stones Trajectory (isometric)", fontsize=10)
