"""
Generate three-panel CFS illustration where optimization variable is full u-trajectory.

This mirrors test/algos/test_cfs_qp_full.py:
- nominal x-reference -> inverse dynamics nominal u
- CFSQPFullFilter optimizes full u trajectory
- rollout filtered u to get "after projection" x-trajectory
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Ellipse, Polygon
from matplotlib.path import Path as MplPath

from genedynamics.envs.single_integrator_box_2d import SingleIntegratorBox2DEnv
from genedynamics.envs.obstacles.base import ObstacleManager
from genedynamics.envs.obstacles.convex import BoxObstacle
from genedynamics.core.constraints.action_filters.cfs_qp_full import CFSQPFullFilter
from genedynamics.core.constraints.core.types import ScheduleState


TRAJ_COLOR = "#1f77b4"
VIOLATION_COLOR = "red"
NUM_POINTS = 18
DT = 0.12
ROBOT_RADIUS = 0.05
GOAL_PENALTY_LAMBDA = 65.0


class _EnvForCFSFull:
    """Thin env wrapper required by CFSQPFullFilter."""

    def __init__(self, base: SingleIntegratorBox2DEnv, robot_radius: float = 0.05):
        self._base = base
        self.dt = float(base.dt)
        self.robot_radius = float(robot_radius)
        self.control_limit = float(getattr(base, "control_limit", 1.0))
        self.p_max = float(getattr(base, "p_max", 2.0))
        self.horizon = int(getattr(base, "horizon", 80))
        self.act_dim = int(getattr(base, "act_dim", 2))

    def model_transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self._base.model_transition(state, action)

    def jax_transition(self, state: Any, action: Any) -> Any:
        return self._base.jax_transition(state, action)


def make_random_obstacle(seed: int = 2) -> np.ndarray:
    """Same obstacle style as visualize_cfs_three_panels.py."""
    rng = np.random.default_rng(seed)
    center = np.array([0.2, 0.2], dtype=np.float32)
    angles = np.sort(rng.uniform(0, 2 * np.pi, size=7))
    radii = rng.uniform(0.40, 0.56, size=angles.shape[0])
    points = np.stack(
        [center[0] + radii * np.cos(angles), center[1] + radii * np.sin(angles)],
        axis=1,
    ).astype(np.float32)
    return points


def make_obstacles(visual_polygon: np.ndarray) -> ObstacleManager:
    """CFS backend uses box obstacles; use polygon AABB to stay consistent."""
    p_min = np.min(visual_polygon, axis=0)
    p_max = np.max(visual_polygon, axis=0)
    center = 0.5 * (p_min + p_max)
    half_extents = 0.5 * (p_max - p_min)

    obstacles = ObstacleManager()
    obstacles.add(
        BoxObstacle(
            center=np.asarray(center, dtype=np.float32),
            half_extents=np.asarray(half_extents, dtype=np.float32),
            name="obstacle_main",
        )
    )
    return obstacles


def get_main_box(obstacles: ObstacleManager) -> BoxObstacle:
    for o in obstacles:
        if hasattr(o, "center") and hasattr(o, "half_extents"):
            return o
    raise RuntimeError("No box obstacle found.")


def x_reference_straight_line(start: np.ndarray, goal: np.ndarray, num_points: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
    return start[None, :] + t[:, None] * (goal - start)[None, :]


def inverse_dynamics_single_integrator(positions: np.ndarray, dt: float) -> np.ndarray:
    diffs = np.diff(positions, axis=0).astype(np.float32)
    return diffs / float(dt)


def rollout_states(x0: np.ndarray, actions: np.ndarray, env: _EnvForCFSFull) -> np.ndarray:
    states = [np.asarray(x0, dtype=np.float32)]
    s = np.asarray(x0, dtype=np.float32)
    for a in actions:
        s = env.model_transition(s, a)
        states.append(np.asarray(s, dtype=np.float32))
    return np.stack(states, axis=0)


def box_signed_distance(point: np.ndarray, center: np.ndarray, half_extents: np.ndarray) -> float:
    """Signed distance to axis-aligned box (negative inside)."""
    q = np.abs(point - center) - half_extents
    outside = np.maximum(q, 0.0)
    outside_dist = float(np.linalg.norm(outside))
    inside_dist = float(np.max(q))
    return outside_dist if outside_dist > 0.0 else inside_dist


def polygon_signed_distance(point: np.ndarray, polygon: np.ndarray) -> float:
    path = MplPath(polygon)
    inside = bool(path.contains_point(point))
    min_dist = math.inf
    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i + 1) % len(polygon)]
        ab = b - a
        t = float(np.dot(point - a, ab) / (np.dot(ab, ab) + 1e-12))
        t = np.clip(t, 0.0, 1.0)
        proj = a + t * ab
        d = float(np.linalg.norm(point - proj))
        if d < min_dist:
            min_dist = d
    return -min_dist if inside else min_dist


def polygon_orientation(polygon: np.ndarray) -> float:
    area = 0.0
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        area += (x1 * y2 - x2 * y1)
    return area


def project_to_polygon_boundary(point: np.ndarray, polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ccw = polygon_orientation(polygon) > 0
    best_proj = point.copy()
    best_normal = np.array([1.0, 0.0], dtype=np.float32)
    min_dist = math.inf
    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i + 1) % len(polygon)]
        ab = b - a
        t = float(np.dot(point - a, ab) / (np.dot(ab, ab) + 1e-12))
        t = np.clip(t, 0.0, 1.0)
        proj = a + t * ab
        d = float(np.linalg.norm(point - proj))
        if d < min_dist:
            min_dist = d
            dx, dy = (b - a)
            if ccw:
                normal = np.array([dy, -dx], dtype=np.float32)
            else:
                normal = np.array([-dy, dx], dtype=np.float32)
            best_proj = proj
            best_normal = normal / (np.linalg.norm(normal) + 1e-12)
    return best_proj, best_normal


def project_to_polygon_boundary_prefer_right(
    point: np.ndarray,
    polygon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Projection variant for visualization: prefer outward normals with +x component
    so the correction is shown on the right side.
    """
    ccw = polygon_orientation(polygon) > 0
    best_proj = point.copy()
    best_normal = np.array([1.0, 0.0], dtype=np.float32)
    best_score = math.inf

    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i + 1) % len(polygon)]
        ab = b - a
        t = float(np.dot(point - a, ab) / (np.dot(ab, ab) + 1e-12))
        t = np.clip(t, 0.0, 1.0)
        proj = a + t * ab
        d = float(np.linalg.norm(point - proj))

        dx, dy = (b - a)
        if ccw:
            normal = np.array([dy, -dx], dtype=np.float32)
        else:
            normal = np.array([-dy, dx], dtype=np.float32)
        normal = normal / (np.linalg.norm(normal) + 1e-12)

        # Strongly penalize non-right-facing normals for panel consistency.
        penalty = 0.0 if normal[0] >= 0.0 else 10.0
        score = d + penalty
        if score < best_score:
            best_score = score
            best_proj = proj
            best_normal = normal

    return best_proj, best_normal


def apply_terminal_penalty(
    u_actions: np.ndarray,
    x0: np.ndarray,
    goal: np.ndarray,
    env: _EnvForCFSFull,
    lam: float,
    tail_steps: int = 8,
) -> np.ndarray:
    """
    Tail-weighted terminal correction for:
      min_u ||u-u0||^2 + lam * ||x_N(u)-goal||^2
    under single-integrator x_N = x0 + dt * sum_t u_t.
    Only last `tail_steps` controls are adjusted to preserve earlier avoidance.
    """
    u = np.asarray(u_actions, dtype=np.float32).copy()
    T = u.shape[0]
    if T <= 0:
        return u

    tail = int(max(1, min(tail_steps, T)))
    w = np.zeros(T, dtype=np.float32)
    w[-tail:] = np.linspace(0.45, 1.0, tail, dtype=np.float32)
    denom_w = float(np.sum(w**2))
    if denom_w < 1e-12:
        return u

    for _ in range(10):
        x_end = rollout_states(x0, u, env)[-1, :2]
        e = x_end - goal
        if float(np.linalg.norm(e)) < 1e-3:
            break
        gain = (lam * env.dt) / (1.0 + lam * (env.dt**2) * denom_w)
        shift = -gain * e
        for t in range(T):
            if w[t] > 0:
                u[t] += w[t] * shift
    return u


def force_terminal_hit(
    u_actions: np.ndarray,
    x0: np.ndarray,
    goal: np.ndarray,
    env: _EnvForCFSFull,
    tail_steps: int = 10,
    iters: int = 12,
) -> np.ndarray:
    """
    Hard terminal correction on tail controls:
    iteratively enforce x_N -> goal while respecting control_limit clipping.
    """
    u = np.asarray(u_actions, dtype=np.float32).copy()
    T = u.shape[0]
    if T <= 0:
        return u
    tail = int(max(1, min(tail_steps, T)))
    w = np.linspace(0.35, 1.0, tail, dtype=np.float32)
    w = w / float(np.sum(w) + 1e-12)
    ctrl_lim = float(getattr(env, "control_limit", 1.0))

    for _ in range(iters):
        x_end = rollout_states(x0, u, env)[-1, :2]
        e = goal - x_end
        if float(np.linalg.norm(e)) < 1e-3:
            break
        shift = e / max(env.dt, 1e-8)
        for j in range(tail):
            idx = T - tail + j
            u[idx] += w[j] * shift
            u[idx] = np.clip(u[idx], -ctrl_lim, ctrl_lim)
    return u


def project_to_inflated_box_boundary(
    point: np.ndarray,
    center: np.ndarray,
    half_extents: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project point to nearest boundary of inflated box and return outward normal."""
    h = half_extents + margin
    d = point - center
    abs_d = np.abs(d)
    dist_to_face = h - abs_d
    axis = int(np.argmin(dist_to_face))
    sign = 1.0 if d[axis] >= 0.0 else -1.0

    proj = point.copy()
    proj[axis] = center[axis] + sign * h[axis]
    other = 1 - axis
    proj[other] = float(np.clip(proj[other], center[other] - h[other], center[other] + h[other]))

    normal = np.zeros(2, dtype=np.float32)
    normal[axis] = sign
    return proj, normal


def clip_polygon_halfspace(
    polygon: list[tuple[float, float]],
    normal: np.ndarray,
    point_on_plane: np.ndarray,
) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clip to keep n·(x-p)>=0."""
    def inside(pt: np.ndarray) -> bool:
        return float(np.dot(normal, pt - point_on_plane)) >= -1e-9

    def intersect(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        d = p2 - p1
        denom = float(np.dot(normal, d))
        if abs(denom) < 1e-12:
            return p1
        t = float(np.dot(normal, point_on_plane - p1)) / denom
        return p1 + t * d

    output: list[tuple[float, float]] = []
    if not polygon:
        return output

    prev = np.array(polygon[-1], dtype=np.float32)
    prev_in = inside(prev)
    for pt in polygon:
        curr = np.array(pt, dtype=np.float32)
        curr_in = inside(curr)
        if curr_in and prev_in:
            output.append(tuple(curr.tolist()))
        elif prev_in and not curr_in:
            output.append(tuple(intersect(prev, curr).tolist()))
        elif not prev_in and curr_in:
            output.append(tuple(intersect(prev, curr).tolist()))
            output.append(tuple(curr.tolist()))
        prev, prev_in = curr, curr_in
    return output


def build_local_convex_set_polygon(
    center: np.ndarray,
    normal: np.ndarray,
    point_on_plane: np.ndarray,
    radius: float = 0.56,
    n_vertices: int = 80,
) -> np.ndarray:
    thetas = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False, dtype=np.float32)
    circle_poly = [(float(center[0] + radius * np.cos(t)), float(center[1] + radius * np.sin(t))) for t in thetas]
    clipped = clip_polygon_halfspace(circle_poly, normal, point_on_plane)
    if not clipped:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(clipped, dtype=np.float32)


def compute_trajectory_data() -> dict[str, Any]:
    start = np.array([-1.0, -1.0], dtype=np.float32)
    goal = np.array([1.0, 1.0], dtype=np.float32)

    visual_polygon = make_random_obstacle(seed=2)
    obstacles = make_obstacles(visual_polygon)
    # Slightly larger control authority helps terminal pull while keeping same CFS-on-u flow.
    base_env = SingleIntegratorBox2DEnv(dt=DT, p_max=2.0, control_limit=1.5)
    env = _EnvForCFSFull(base_env, robot_radius=ROBOT_RADIUS)

    x_ref = x_reference_straight_line(start, goal, NUM_POINTS)
    u_nom = inverse_dynamics_single_integrator(x_ref, DT)
    x_nom = rollout_states(x_ref[0], u_nom, env)[:, :2]

    filtr = CFSQPFullFilter(
        max_constraints_per_point=8,
        constraint_margin=0.25,
        use_slack=True,
    )
    sched_state = ScheduleState(k=0, K=1)
    sched_params = {
        "margin": 0.1,
        "rho": 10.0,
        "qp_gate": True,
        "qp_prob": 1.0,
        "I_QP": 8,
        "cfs_outer_iters": 8,
    }
    u_filtered = filtr.apply_actions(
        x_ref[0],
        u_nom,
        env=env,
        obstacles=obstacles,
        schedule_state=sched_state,
        schedule_params=sched_params,
    )
    u_filtered = np.asarray(u_filtered, dtype=np.float32)
    x_filtered_only = rollout_states(x_ref[0], u_filtered, env)[:, :2]

    # Use displayed obstacle geometry for red violation markers.
    sdf_nom = np.array([polygon_signed_distance(p, visual_polygon) for p in x_nom], dtype=np.float32)
    v_nom = sdf_nom < ROBOT_RADIUS
    # Choose panel-2 linearization point as the 1st violating point on nominal traj.
    viol_idx = np.where(v_nom)[0]
    if len(viol_idx) > 0:
        idx_proj = int(viol_idx[0])
    else:
        idx_proj = int(np.argmin(sdf_nom))
    xk = x_nom[idx_proj]
    proj, normal = project_to_polygon_boundary_prefer_right(xk, visual_polygon)
    proj = proj + normal * ROBOT_RADIUS
    sdf_filtered_only = np.array(
        [polygon_signed_distance(p, visual_polygon) for p in x_filtered_only], dtype=np.float32
    )
    v_filtered_only = sdf_filtered_only < ROBOT_RADIUS

    def evaluate_traj(x_traj: np.ndarray) -> tuple[int, int, float]:
        sdf_vals = np.array([polygon_signed_distance(p, visual_polygon) for p in x_traj], dtype=np.float32)
        violations = int(np.sum(sdf_vals < ROBOT_RADIUS))
        lo = max(0, idx_proj - 5)
        hi = min(len(x_traj), idx_proj + 7)
        side_vals = [float(np.dot(normal, x_traj[i] - proj)) for i in range(lo, hi)]
        side_viol = int(np.sum(np.array(side_vals, dtype=np.float32) < -1e-4))
        end_dist = float(np.linalg.norm(x_traj[-1] - goal))
        return violations, side_viol, end_dist

    best_x = x_filtered_only
    best_u = u_filtered
    best_sdf = sdf_filtered_only
    best_v = v_filtered_only
    best_score = evaluate_traj(best_x)

    # Try stronger terminal pulls and re-run CFS; choose lexicographically by
    # (violations, side-consistency wrt panel2, end distance).
    for lam in [GOAL_PENALTY_LAMBDA, 120.0, 220.0, 380.0, 700.0, 1200.0]:
        for tail in [8, 12, 16]:
            u_goal = apply_terminal_penalty(u_filtered, x_ref[0], goal, env, lam=lam, tail_steps=tail)
            refine_params = dict(sched_params)
            refine_params["I_QP"] = 18
            refine_params["cfs_outer_iters"] = 18
            u_refined = filtr.apply_actions(
                x_ref[0],
                u_goal,
                env=env,
                obstacles=obstacles,
                schedule_state=sched_state,
                schedule_params=refine_params,
            )
            u_refined = np.asarray(u_refined, dtype=np.float32)
            x_candidate = rollout_states(x_ref[0], u_refined, env)[:, :2]
            score = evaluate_traj(x_candidate)
            if score < best_score:
                best_score = score
                best_x = x_candidate
                best_u = u_refined
                best_sdf = np.array([polygon_signed_distance(p, visual_polygon) for p in x_candidate], dtype=np.float32)
                best_v = best_sdf < ROBOT_RADIUS

    # Final hard terminal correction on selected u-candidate.
    u_terminal = force_terminal_hit(best_u, x_ref[0], goal, env, tail_steps=10, iters=14)
    x_terminal = rollout_states(x_ref[0], u_terminal, env)[:, :2]
    sdf_terminal = np.array([polygon_signed_distance(p, visual_polygon) for p in x_terminal], dtype=np.float32)
    v_terminal = sdf_terminal < ROBOT_RADIUS

    # Prefer terminal-corrected trajectory unless it introduces more collisions.
    if int(np.sum(v_terminal)) <= int(np.sum(best_v)):
        x_after = x_terminal
        sdf_after = sdf_terminal
        v_after = v_terminal
    else:
        x_after = best_x
        sdf_after = best_sdf
        v_after = best_v

    # Keep panel-2 explicitly right-side as requested.
    if normal[0] < 0.0:
        normal = -normal

    return {
        "start": start,
        "goal": goal,
        "x_nom": x_nom,
        "x_after": x_after,
        "sdf_nom": sdf_nom,
        "sdf_after": sdf_after,
        "v_nom": v_nom,
        "v_after": v_after,
        "obstacles": obstacles,
        "visual_polygon": visual_polygon,
        "proj_idx": idx_proj,
        "proj_point": proj,
        "proj_normal": normal,
    }


def draw_start_goal(ax: plt.Axes, start: np.ndarray, goal: np.ndarray, robot_radius: float) -> None:
    ax.add_patch(
        Circle(
            (float(start[0]), float(start[1])),
            radius=robot_radius,
            facecolor="green",
            alpha=0.6,
            edgecolor="darkgreen",
            linewidth=1.5,
            zorder=10,
        )
    )
    ax.scatter([start[0]], [start[1]], c="darkgreen", s=12, marker="o", zorder=11)

    ax.add_patch(
        Circle(
            (float(goal[0]), float(goal[1])),
            radius=robot_radius,
            facecolor="none",
            edgecolor="red",
            linewidth=1.5,
            zorder=10,
        )
    )
    ax.plot(goal[0], goal[1], "r*", markersize=6, zorder=11)


def draw_obstacle(ax: plt.Axes, visual_polygon: np.ndarray, filled: bool) -> None:
    ax.add_patch(
        Polygon(
            visual_polygon,
            closed=True,
            fill=filled,
            facecolor="#c8c8c8",
            edgecolor="black",
            linewidth=1.2,
            alpha=0.9 if filled else 0.35,
            zorder=1,
        )
    )


def style_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_nominal(ax: plt.Axes, data: dict[str, Any]) -> None:
    x_nom = data["x_nom"]
    v_nom = data["v_nom"]

    segments = np.stack([x_nom[:-1], x_nom[1:]], axis=1)
    ax.add_collection(LineCollection(segments, colors=TRAJ_COLOR, linewidths=2.5, zorder=3))
    ax.scatter(x_nom[:, 0], x_nom[:, 1], s=24, color=TRAJ_COLOR, zorder=4)
    if np.any(v_nom):
        vv = x_nom[v_nom]
        ax.scatter(vv[:, 0], vv[:, 1], s=34, color=VIOLATION_COLOR, zorder=5)

    draw_obstacle(ax, data["visual_polygon"], filled=True)
    draw_start_goal(ax, data["start"], data["goal"], ROBOT_RADIUS)
    style_axis(ax)


def panel_projection_geometry(ax: plt.Axes, data: dict[str, Any]) -> None:
    x_nom = data["x_nom"]
    v_nom = data["v_nom"]
    obstacles = data["obstacles"]
    visual_polygon = data["visual_polygon"]
    sdf = np.array([polygon_signed_distance(p, visual_polygon) for p in x_nom], dtype=np.float32)
    idx = int(data["proj_idx"])
    xk = x_nom[idx]
    proj = np.asarray(data["proj_point"], dtype=np.float32)
    normal = np.asarray(data["proj_normal"], dtype=np.float32)

    # Shift local trust-region center to projected feasible side to avoid
    # visual overlap that looks like "feasible set on obstacle".
    f_center = proj + 0.22 * normal
    f_poly = build_local_convex_set_polygon(
        center=f_center,
        normal=normal,
        point_on_plane=proj,
        radius=0.46,
        n_vertices=80,
    )
    if len(f_poly) > 0:
        ax.add_patch(
            Polygon(
                f_poly,
                closed=True,
                fill=True,
                facecolor="#9ecae1",
                edgecolor="#6baed6",
                linewidth=1.0,
                alpha=0.20,
                zorder=0,
            )
        )

    draw_obstacle(ax, visual_polygon, filled=False)
    ax.plot(x_nom[:, 0], x_nom[:, 1], color=TRAJ_COLOR, linewidth=2.2, zorder=2)
    ax.scatter(x_nom[:, 0], x_nom[:, 1], s=24, color=TRAJ_COLOR, zorder=2)
    if np.any(v_nom):
        vv = x_nom[v_nom]
        ax.scatter(vv[:, 0], vv[:, 1], s=34, color=VIOLATION_COLOR, zorder=3)

    ax.scatter([xk[0]], [xk[1]], s=70, color=VIOLATION_COLOR, zorder=4)
    ax.scatter([proj[0]], [proj[1]], s=86, color=TRAJ_COLOR, edgecolors="black", linewidths=0.9, zorder=6)
    ax.annotate("", xy=proj, xytext=xk, arrowprops=dict(arrowstyle="->", color="black", linewidth=2), zorder=4)

    tangent = np.array([-normal[1], normal[0]], dtype=np.float32)
    p1 = proj - tangent * 1.1
    p2 = proj + tangent * 1.1
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle="--", color="#999999", linewidth=1.0, zorder=1)

    delta = proj - xk
    dist = float(np.linalg.norm(delta))
    theta = float(np.degrees(np.arctan2(delta[1], delta[0]))) if dist > 1e-9 else 45.0
    a_mid = max(dist, 0.12)
    b_mid = 0.45 * a_mid
    for scale, alpha in [(0.70, 0.55), (1.00, 0.42), (1.35, 0.30)]:
        ax.add_patch(
            Ellipse(
                xy=(float(xk[0]), float(xk[1])),
                width=2.0 * a_mid * scale,
                height=2.0 * b_mid * scale,
                angle=theta,
                fill=False,
                edgecolor="#9a9a9a",
                linestyle="-",
                linewidth=0.9,
                alpha=alpha,
                zorder=2,
            )
        )
    style_axis(ax)


def panel_after(ax: plt.Axes, data: dict[str, Any]) -> None:
    x_after = data["x_after"]
    v_after = data["v_after"]

    draw_obstacle(ax, data["visual_polygon"], filled=True)
    ax.plot(x_after[:, 0], x_after[:, 1], color=TRAJ_COLOR, linewidth=2.5, zorder=3)
    ax.scatter(x_after[:, 0], x_after[:, 1], s=24, color=TRAJ_COLOR, zorder=4)
    if np.any(v_after):
        vv = x_after[v_after]
        ax.scatter(vv[:, 0], vv[:, 1], s=34, color=VIOLATION_COLOR, zorder=5)
    draw_start_goal(ax, data["start"], data["goal"], ROBOT_RADIUS)
    style_axis(ax)


def main() -> None:
    data = compute_trajectory_data()
    out_dir = Path("results/cfs_panels_u")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panel_nominal(axes[0], data)
    panel_projection_geometry(axes[1], data)
    panel_after(axes[2], data)
    plt.tight_layout()
    fig.savefig(out_dir / "cfs_three_panels_u.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    for idx, name in enumerate(["panel1_nominal_u.png", "panel2_projection_u.png", "panel3_after_u.png"]):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        if idx == 0:
            panel_nominal(ax, data)
        elif idx == 1:
            panel_projection_geometry(ax, data)
        else:
            panel_after(ax, data)
        plt.tight_layout()
        fig.savefig(out_dir / name, dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved: {out_dir / 'cfs_three_panels_u.png'}")
    print(f"Saved: {out_dir / 'panel1_nominal_u.png'}")
    print(f"Saved: {out_dir / 'panel2_projection_u.png'}")
    print(f"Saved: {out_dir / 'panel3_after_u.png'}")


if __name__ == "__main__":
    main()
