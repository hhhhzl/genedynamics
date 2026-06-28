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


def _draw_corridor_scene(ax: plt.Axes, scene: SceneIR, *, show_labels: bool = True) -> None:
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

    # Boxes. Each non-box obstacle (sphere/qc) also carries an AABB BoxIR so
    # legacy code can find x_min/x_max; skip drawing the rectangle for those
    # (the sphere/qc patch above is the real shape) to avoid a box ghost.
    for b in scene.boxes:
        cx_b = 0.5 * (b.x_min + b.x_max)
        cy_b = 0.5 * (b.y_min + b.y_max)
        if b.shape == "box":
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
        if show_labels and w_obs > 0.15 and h_obs > 0.06:
            ax.text(cx_b, cy_b, b.name, fontsize=4, ha="center", va="center",
                    color="white", zorder=4, clip_on=True)

    # Start / Goal markers
    if scene.start is not None:
        ax.plot(scene.start[0], scene.start[1], "o", color="#38a169",
                markersize=10, zorder=10, label="start")
    if scene.goal is not None:
        ax.plot(scene.goal[0], scene.goal[1], "*", color="#e53e3e",
                markersize=14, zorder=10, label="goal")


def _articulated_arm_floor(a_tuck: float, posture: float, side_sign: float):
    """Shoulder->elbow->hand offsets RELATIVE TO THE SHOULDER, in the body
    floor frame (fwd = +heading, lat: +side_sign = this arm's outboard side).

    Mirrors the follower's G1 arm behaviour in spirit while staying BOUNDED so
    the 2D top-down hand never pokes past the body envelope (the planner only
    constrains a lateral arm point and clears it in 3D; an unbounded forward
    fold would draw a collision that neither the plan nor the follower has):
      - posture p (shoulder pitch) and tuck a swing the forearm FORWARD;
      - open arms (a->0) bow the elbow slightly OUTBOARD (~0.05, the planner's
        arm-protrusion budget); tucked arms (a->1) fold the hand INBOARD to the
        chest centerline, exactly the squeeze posture.
    Returns (shoulder, elbow, hand) offsets as (fwd, lat) tuples.
    """
    a = float(np.clip(a_tuck, 0.0, 1.0))
    p = float(np.clip(posture, -1.0, 1.0))
    fwd_drive = max(p, 0.0)
    # Elbow: just forward of the shoulder, slightly outboard when arms are open.
    e_fwd = 0.04 + 0.03 * fwd_drive
    e_lat = side_sign * 0.05 * (1.0 - a)
    # Hand: forward to ~the chest front, folding inboard as the arm tucks.
    h_fwd = 0.09 + 0.04 * a + 0.03 * fwd_drive
    h_lat = side_sign * (0.02 - 0.18 * a)
    return (0.0, 0.0), (e_fwd, e_lat), (h_fwd, h_lat)


def _draw_body_ghost(
    ax: plt.Axes,
    pose: RobotPoseIR,
    *,
    torso_a: float,
    torso_b: float,
    torso_crouch_extra: float,
    h_nominal: float,
    arm_reach_open: float,
    arm_reach_tucked: float,
    body_shape: str = "ellipse",
    arm_style: str = "articulated",
    face_color: Any = None,
    show_arrow: bool = True,
    edge_color: str = "#2c5282",
    linewidth: float = 0.6,
    alpha: float = 0.4,
    zorder: float = 8,
) -> None:
    """Draw one humanoid pose, top-down, as a SHOULDER-WIDE body.

    The torso is drawn 2*TORSO_A=0.40 wide LATERALLY (shoulders) and
    2*TORSO_B=0.22 deep FORWARD — the convention the corridor's binding
    constraints use (wall clearance + W_eff squeeze both treat TORSO_A as the
    lateral half-width), and the one that matches the real G1 / the follower.
    On top of the torso footprint we add a shoulder bar, a head disc, a heading
    arrow, and two arms (heading = yaw + torso_yaw, so a base-yaw turn swings
    the whole body). Body fill is ``face_color`` (time/gray), not height.
    """
    heading = pose.yaw + pose.torso_yaw
    # Lateral (shoulder) half-width grows a touch when crouching; forward
    # (chest) half-depth is the short axis.
    lat_half = torso_a + torso_crouch_extra * max(0.0, h_nominal - pose.body_height)
    fwd_half = torso_b
    if face_color is None:
        face_color = (0.45, 0.5, 0.55, 1.0)

    c, s = np.cos(heading), np.sin(heading)

    def _to_world(fwd, lat):
        # fwd = +heading, lat = +robot-left; same rotation as _arm_pos_np.
        return (pose.x + c * fwd - s * lat, pose.y + s * fwd + c * lat)

    # --- Torso footprint: WIDE laterally (shoulders), shallow forward ---
    if body_shape == "capsule":
        # Stadium whose long axis is LATERAL (shoulder span), built in the
        # body floor frame then rotated. Explicit polygon avoids
        # FancyBboxPatch's display-space rounding collapsing under rotation.
        straight = max(lat_half - fwd_half, 1e-3)  # half-length of flat part
        th = np.linspace(-np.pi / 2, np.pi / 2, 16)
        cap_top = np.column_stack((fwd_half * np.sin(th),
                                   straight + fwd_half * np.cos(th)))   # +lat cap
        cap_bot = np.column_stack((-fwd_half * np.sin(th),
                                   -straight - fwd_half * np.cos(th)))  # -lat cap
        loc = np.vstack((cap_top, cap_bot))  # columns: (fwd, lat)
        world = np.column_stack((
            pose.x + c * loc[:, 0] - s * loc[:, 1],
            pose.y + s * loc[:, 0] + c * loc[:, 1],
        ))
        ax.add_patch(mpatches.Polygon(
            world, closed=True, facecolor=face_color, edgecolor=edge_color,
            alpha=alpha, linewidth=linewidth, zorder=zorder,
        ))
    else:  # "ellipse"
        ax.add_patch(mpatches.Ellipse(
            (pose.x, pose.y),
            width=2 * fwd_half,     # forward (along heading), 0.22
            height=2 * lat_half,    # lateral (shoulders), 0.40
            angle=np.degrees(heading),
            facecolor=face_color, edgecolor=edge_color,
            alpha=alpha, linewidth=linewidth, zorder=zorder,
        ))

    # --- Shoulder bar across the lateral (shoulder) axis ---
    shoulder_half = lat_half * 0.92
    sL = _to_world(0.0, shoulder_half)
    sR = _to_world(0.0, -shoulder_half)
    ax.plot([sL[0], sR[0]], [sL[1], sR[1]], color=edge_color,
            linewidth=linewidth + 1.2, alpha=min(1.0, alpha + 0.15),
            solid_capstyle="round", zorder=zorder + 0.3)

    # --- Head: small disc just forward of the shoulder line ---
    hd = _to_world(0.30 * fwd_half, 0.0)
    ax.add_patch(mpatches.Circle(
        hd, 0.055, facecolor=edge_color, edgecolor="none",
        alpha=min(1.0, alpha + 0.2), zorder=zorder + 0.35,
    ))

    # --- Heading arrow ---
    if show_arrow:
        tip = _to_world(fwd_half + 0.10, 0.0)
        ax.annotate(
            "", xy=tip, xytext=(pose.x, pose.y),
            arrowprops=dict(
                arrowstyle="->,head_length=0.4,head_width=0.25",
                color="#1a202c", lw=1.0, alpha=min(1.0, alpha + 0.3),
                shrinkA=0.0, shrinkB=0.0,
            ),
            zorder=zorder + 0.5,
        )

    # --- Arms from the shoulder-bar ends ---
    arm_alpha = min(1.0, alpha * 0.9 + 0.1)
    for sign, a_tuck, posture, color in [
        (1.0, pose.arm_tuck_L, pose.arm_posture_L, "#e53e3e"),
        (-1.0, pose.arm_tuck_R, pose.arm_posture_R, "#dd6b20"),
    ]:
        sh_lat = sign * shoulder_half
        sw = _to_world(0.0, sh_lat)
        if arm_style == "articulated":
            # follower-faithful shoulder->elbow->hand, BOUNDED (offsets are
            # relative to the shoulder; hands fold to the chest front).
            (_, _), (ef, el), (hf, hl) = _articulated_arm_floor(a_tuck, posture, sign)
            ew = _to_world(ef, sh_lat + el)
            hw = _to_world(hf, sh_lat + hl)
            lw = 2.0 if a_tuck > 0.5 else 1.4
            ax.plot([sw[0], ew[0], hw[0]], [sw[1], ew[1], hw[1]],
                    color=color, linewidth=lw, alpha=arm_alpha,
                    zorder=zorder + 0.1, solid_capstyle="round",
                    solid_joinstyle="round")
            ax.plot(ew[0], ew[1], "o", color=color, markersize=2.0,
                    alpha=arm_alpha, zorder=zorder + 0.15)  # elbow
            ax.plot(hw[0], hw[1], "o", color=color,
                    markersize=4.0 if a_tuck > 0.5 else 3.0,
                    alpha=arm_alpha, zorder=zorder + 0.2)    # hand
        else:  # "point": planner's lateral arm (center -> lateral reach)
            reach = arm_reach_open + (arm_reach_tucked - arm_reach_open) * np.clip(a_tuck, 0, 1)
            hw = _to_world(0.0, sign * max(reach, 0.06))
            lw = 2.0 if a_tuck > 0.5 else 1.2
            ax.plot([sw[0], hw[0]], [sw[1], hw[1]], color=color, linewidth=lw,
                    alpha=arm_alpha, zorder=zorder + 0.1, solid_capstyle="round")
            ax.plot(hw[0], hw[1], "o", color=color,
                    markersize=4.0 if a_tuck > 0.5 else 2.5,
                    alpha=arm_alpha, zorder=zorder + 0.2)


def _scene_clearance(pose: RobotPoseIR, scene: SceneIR) -> float:
    """Env-free min distance from the body center to walls + obstacles.

    A geometric proxy (no env import) used only to pick the most
    informative frame to highlight, not for any safety claim.
    """
    _, _, ymin, ymax = scene.bounds
    d = min(pose.y - ymin, ymax - pose.y)
    for s in scene.spheres:
        d = min(d, float(np.hypot(pose.x - s.cx, pose.y - s.cy) - s.radius))
    for qc in scene.quarter_circles:
        d = min(d, float(np.hypot(pose.x - qc.cx, pose.y - qc.cy) - qc.radius))
    for b in scene.boxes:
        dx = max(b.x_min - pose.x, 0.0, pose.x - b.x_max)
        dy = max(b.y_min - pose.y, 0.0, pose.y - b.y_max)
        d = min(d, float(np.hypot(dx, dy)))
    return d


def _pick_highlight(
    poses: List[RobotPoseIR],
    scene: SceneIR,
    highlight: str,
    coll_arr: Optional[np.ndarray],
) -> Optional[int]:
    """Choose the frame to emphasize. Precedence: collision > narrowest."""
    T = len(poses)
    if T == 0 or highlight == "none":
        return None
    if highlight == "final":
        return T - 1
    # "narrowest": prefer the first colliding frame if any, else min-clearance.
    if coll_arr is not None and bool(coll_arr.any()):
        return int(np.argmax(coll_arr))
    return min(range(T), key=lambda t: _scene_clearance(poses[t], scene))


def render_corridor_top(
    fig: Any,
    ax: plt.Axes,
    scene: SceneIR,
    poses: List[RobotPoseIR],
    extras: Optional[Dict[str, Any]] = None,
) -> None:
    """Render corridor scene + body ghosts (publication top-down style).

    Required ``extras`` keys:
      - torso_a, torso_b, torso_crouch_extra
      - h_nominal, arm_reach_open, arm_reach_tucked
    Optional ``extras`` keys:
      - collision_mask (bool list/array, length T)
      - h_min, h_max (only used when show_colorbar=True)
    Paper-figure knobs (all optional, defaults shown):
      - body_shape: "ellipse" | "capsule"        (default "ellipse")
      - ghost_color: "time" | "gray"             (default "time")
      - ghost_cmap: matplotlib cmap name         (default "Blues")
      - highlight: "narrowest" | "final" | "none" (default "narrowest")
      - n_ghosts: int                            (default 9)
      - ghost_alpha_min, ghost_alpha_max: float  (default 0.25, 0.6)
      - show_heading_arrows: bool                (default True)
      - show_colorbar: bool                      (default False)
    """
    extras = extras or {}
    _draw_corridor_scene(ax, scene,
                         show_labels=bool(extras.get("show_obstacle_labels", False)))

    if not poses:
        return

    body_shape = extras.get("body_shape", "ellipse")
    arm_style = extras.get("arm_style", "articulated")
    ghost_color = extras.get("ghost_color", "time")
    ghost_cmap = extras.get("ghost_cmap", "Blues")
    highlight = extras.get("highlight", "narrowest")
    show_arrows = bool(extras.get("show_heading_arrows", True))
    show_cbar = bool(extras.get("show_colorbar", False))
    n_ghosts = int(extras.get("n_ghosts", 9))
    a_lo = float(extras.get("ghost_alpha_min", 0.25))
    a_hi = float(extras.get("ghost_alpha_max", 0.6))
    # Publication-style chrome (paper defaults: big fonts, no title/legend/labels).
    axis_fontsize = float(extras.get("axis_fontsize", 26))
    tick_fontsize = float(extras.get("tick_fontsize", 22))
    show_title = bool(extras.get("show_title", False))
    show_legend = bool(extras.get("show_legend", False))
    # mode_paths: optional list of (T,2) xy arrays = other candidate trajectories
    # (multi-modal). Drawn as thin faded lines under the selected mode's bodies.
    mode_paths = extras.get("mode_paths")

    xs = np.array([p.x for p in poses], dtype=np.float32)
    ys = np.array([p.y for p in poses], dtype=np.float32)
    T = xs.shape[0]

    # Animation: when partial_until_step is set the figure is one GIF frame —
    # draw the trajectory/ghosts only up to that step (a growing reveal) and
    # highlight the CURRENT pose instead of the min-clearance one.
    partial_t = extras.get("partial_until_step")
    animating = partial_t is not None
    T_vis = min(int(partial_t) + 1, T) if animating else T

    body_kw = dict(
        torso_a=extras["torso_a"], torso_b=extras["torso_b"],
        torso_crouch_extra=extras["torso_crouch_extra"],
        h_nominal=extras["h_nominal"],
        arm_reach_open=extras["arm_reach_open"],
        arm_reach_tucked=extras["arm_reach_tucked"],
        body_shape=body_shape,
        arm_style=arm_style,
    )

    # Multi-modal: other candidate trajectories as thin faded centerlines
    # (revealed up to the current step when animating).
    if mode_paths:
        mcmap = cm.get_cmap("tab10")
        for mi, path in enumerate(mode_paths):
            p = np.asarray(path, dtype=np.float32).reshape(-1, 2)[:T_vis]
            if p.shape[0] < 2:
                continue
            ax.plot(p[:, 0], p[:, 1], "-", color=mcmap(mi % 10),
                    linewidth=1.2, alpha=0.45, zorder=4.5,
                    label=f"mode {mi}" if show_legend else None)

    # Selected trajectory centerline (revealed up to the current step).
    ax.plot(xs[:T_vis], ys[:T_vis], "-", color="#3182ce", linewidth=1.8,
            alpha=0.5, zorder=6, label="trajectory")

    # Equally-spaced sampled ghost frames within the visible window.
    if T_vis <= n_ghosts:
        idxs = list(range(T_vis))
    else:
        idxs = sorted(set(np.linspace(0, T_vis - 1, n_ghosts).round().astype(int).tolist()))

    cmap = cm.get_cmap(ghost_cmap)
    for t in idxs:
        frac = t / max(T - 1, 1)
        a = a_lo + (a_hi - a_lo) * frac
        if ghost_color == "time":
            rgba = cmap(0.25 + 0.65 * frac)  # skip the near-white low end
            fc = (rgba[0], rgba[1], rgba[2], 1.0)
        else:
            fc = (0.45, 0.5, 0.55, 1.0)
        _draw_body_ghost(ax, poses[t], face_color=fc, show_arrow=show_arrows,
                         alpha=a, zorder=6 + 0.01 * t, **body_kw)

    # Highlight ONE key frame on top (amber): current pose while animating,
    # else the min-clearance / collision frame.
    coll = extras.get("collision_mask")
    coll_arr = None
    if coll is not None:
        coll_arr = np.asarray(coll).astype(bool).reshape(-1)

    if animating:
        key_idx = T_vis - 1 if T_vis > 0 else None
    else:
        key_idx = _pick_highlight(poses, scene, highlight, coll_arr)
    if key_idx is not None:
        _draw_body_ghost(ax, poses[key_idx],
                         face_color=(0.95, 0.77, 0.06, 1.0),
                         edge_color="#7c2d12", linewidth=1.4,
                         show_arrow=True, alpha=0.95, zorder=15, **body_kw)
        ax.plot(poses[key_idx].x, poses[key_idx].y, marker="o",
                color="#7c2d12", markersize=3, zorder=15.5)

    # Collision marks (only within the visible window when animating)
    if coll_arr is not None:
        for t in range(min(T_vis, coll_arr.shape[0])):
            if coll_arr[t]:
                ax.plot(xs[t], ys[t], "x", color="red",
                        markersize=8, zorder=20)

    # Optional height colorbar (off by default for the paper figure).
    if show_cbar and "h_min" in extras and "h_max" in extras:
        sm = plt.cm.ScalarMappable(
            cmap="coolwarm_r",
            norm=plt.Normalize(vmin=extras["h_min"], vmax=extras["h_max"]),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=1.0, pad=0.02, aspect=30)
        cbar.set_label("Body height (m)", fontsize=7)

    # Legend (off by default for the paper figure)
    if show_legend:
        legend_patches = [
            mpatches.Patch(color=_Z_COLOR["full"][0], alpha=_Z_COLOR["full"][1], label="Full-height"),
            mpatches.Patch(color=_Z_COLOR["low"][0], alpha=_Z_COLOR["low"][1], label="Low bar (duck)"),
            mpatches.Patch(color=_Z_COLOR["floor"][0], alpha=_Z_COLOR["floor"][1], label="Floor block"),
            mpatches.Patch(color=_Z_COLOR["mid"][0], alpha=_Z_COLOR["mid"][1], label="Aerial (sidestep)"),
        ]
        if key_idx is not None:
            label = ("collision frame" if (coll_arr is not None and key_idx < coll_arr.shape[0]
                                           and coll_arr[key_idx])
                     else "min-clearance frame")
            legend_patches.append(
                mpatches.Patch(color=(0.95, 0.77, 0.06), label=label))
        ax.legend(handles=legend_patches, loc="upper left", fontsize=14, framealpha=0.7)

    ax.set_xlabel("x (m)", fontsize=axis_fontsize)
    ax.set_ylabel("y (m)", fontsize=axis_fontsize)
    if show_title:
        ax.set_title("Corridor trajectory (top-down)", fontsize=max(axis_fontsize - 2, 10))
    ax.tick_params(labelsize=tick_fontsize)


# ---------------------------------------------------------------------------
# Stepping stones
# ---------------------------------------------------------------------------
def draw_stepping_scene(ax: plt.Axes, scene: SceneIR) -> None:
    """Public helper: draw a stepping-stones scene background only."""
    _draw_stepping_scene(ax, scene)


def draw_corridor_scene(ax: plt.Axes, scene: SceneIR, *, show_labels: bool = False) -> None:
    """Public helper: draw a corridor scene background only.

    Labels OFF by default for publication consistency (used by the diffusion-
    steps viz via stepping_stones_trajectory._draw_scene_ir); the main
    trajectory render gates labels via the show_obstacle_labels config knob.
    """
    _draw_corridor_scene(ax, scene, show_labels=show_labels)


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
    axis_fontsize = float(extras.get("axis_fontsize", 26))
    tick_fontsize = float(extras.get("tick_fontsize", 22))
    _draw_stepping_scene(ax, scene)

    if not poses:
        ax.set_xlabel("x (m)", fontsize=axis_fontsize)
        ax.set_ylabel("y (m)", fontsize=axis_fontsize)
        ax.tick_params(labelsize=tick_fontsize)
        ax.grid(False)
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
        # End/goal star: match the diffusion_steps figures (red '*', plot markersize=15 ~ scatter s=225)
        ax.scatter(body[-1, 0], body[-1, 1], marker="*",
                   s=225, color="r", zorder=10)

    ax.set_xlabel("x (m)", fontsize=axis_fontsize)
    ax.set_ylabel("y (m)", fontsize=axis_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    ax.grid(False)
