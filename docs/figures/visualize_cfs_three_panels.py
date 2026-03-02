"""
Generate three-panel CFS illustration figure.

Panels:
1) Nominal trajectory (violates obstacle)
2) Projection operator geometry (local convex set)
3) After projection (feasible trajectory)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Ellipse, Polygon
from matplotlib.path import Path as MplPath


TRAJ_COLOR = "#1f77b4"
VIOLATION_COLOR = "red"
NUM_POINTS = 18


def make_random_obstacle(seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    center = np.array([0.2, 0.2], dtype=np.float32)
    angles = np.sort(rng.uniform(0, 2 * np.pi, size=7))
    radii = rng.uniform(0.40, 0.56, size=angles.shape[0])
    points = np.stack(
        [center[0] + radii * np.cos(angles), center[1] + radii * np.sin(angles)],
        axis=1,
    ).astype(np.float32)
    return points


def x_reference_straight_line(
    start: np.ndarray,
    goal: np.ndarray,
    num_points: int,
) -> np.ndarray:
    t = np.linspace(0, 1, num_points, dtype=np.float32)
    return start[None, :] + t[:, None] * (goal - start)[None, :]


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


def polygon_centroid(polygon: np.ndarray) -> np.ndarray:
    return np.mean(polygon, axis=0)


def inflate_polygon(polygon: np.ndarray, margin: float) -> np.ndarray:
    center = polygon_centroid(polygon)
    inflated = []
    for v in polygon:
        d = v - center
        n = np.linalg.norm(d) + 1e-12
        inflated.append(v + (d / n) * margin)
    return np.asarray(inflated, dtype=np.float32)


def segment_min_sdf(a: np.ndarray, b: np.ndarray, polygon: np.ndarray, samples: int = 80) -> float:
    ts = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    vals = []
    for t in ts:
        p = (1.0 - t) * a + t * b
        vals.append(polygon_signed_distance(p, polygon))
    return float(np.min(vals))


def resample_polyline(points: np.ndarray, num_points: int) -> np.ndarray:
    if len(points) == num_points:
        return points.astype(np.float32)
    seg_lens = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])
    if total < 1e-12:
        return np.repeat(points[:1], num_points, axis=0).astype(np.float32)
    targets = np.linspace(0.0, total, num_points, dtype=np.float32)
    out = []
    j = 0
    for d in targets:
        while j + 1 < len(cum) and cum[j + 1] < d:
            j += 1
        if j + 1 >= len(cum):
            out.append(points[-1].copy())
            continue
        w = (d - cum[j]) / max(cum[j + 1] - cum[j], 1e-12)
        out.append((1.0 - w) * points[j] + w * points[j + 1])
    return np.asarray(out, dtype=np.float32)


def polygon_orientation(polygon: np.ndarray) -> float:
    area = 0.0
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        area += (x1 * y2 - x2 * y1)
    return area


def project_to_polygon_boundary(
    point: np.ndarray,
    polygon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project point to closest polygon edge.
    Returns (projected_point, outward_normal).
    """
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
            n = normal / (np.linalg.norm(normal) + 1e-12)
            best_proj = proj
            best_normal = n
    return best_proj, best_normal


def compute_projection_geometry(
    points: np.ndarray,
    polygon: np.ndarray,
    robot_radius: float,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
    sdf = np.array([polygon_signed_distance(p, polygon) for p in points], dtype=np.float32)
    idx = int(np.argmin(sdf))
    nominal_pt = points[idx]
    proj, normal = project_to_polygon_boundary(nominal_pt, polygon)
    proj = proj + normal * robot_radius
    return sdf, idx, nominal_pt, proj, normal


def clip_polygon_halfspace(
    polygon: list[tuple[float, float]],
    normal: np.ndarray,
    point_on_plane: np.ndarray,
) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clip to keep points with n·(x-p)>=0."""
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
            inter = intersect(prev, curr)
            output.append(tuple(inter.tolist()))
        elif not prev_in and curr_in:
            inter = intersect(prev, curr)
            output.append(tuple(inter.tolist()))
            output.append(tuple(curr.tolist()))
        prev, prev_in = curr, curr_in
    return output


def build_local_convex_set_polygon(
    center: np.ndarray,
    normal: np.ndarray,
    point_on_plane: np.ndarray,
    radius: float = 0.52,
    n_vertices: int = 80,
) -> np.ndarray:
    """
    Approximate local CFS set as:
      F(x^(k)) = {x : n^T(x - p) >= 0} ∩ {x : ||x - center||_2 <= radius}
    represented by a clipped polygon.
    """
    thetas = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False, dtype=np.float32)
    circle_poly = [(float(center[0] + radius * np.cos(t)), float(center[1] + radius * np.sin(t))) for t in thetas]
    clipped = clip_polygon_halfspace(circle_poly, normal, point_on_plane)
    if not clipped:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(clipped, dtype=np.float32)


def draw_obstacle(ax: plt.Axes, polygon: np.ndarray, filled: bool = True) -> None:
    ax.add_patch(
        Polygon(
            polygon,
            closed=True,
            fill=filled,
            facecolor="#c8c8c8",
            edgecolor="black",
            linewidth=1.2,
            hatch=None,
            alpha=0.9 if filled else 0.35,
            zorder=1,
        )
    )


def draw_start_goal(ax: plt.Axes, start: np.ndarray, goal: np.ndarray, robot_radius: float) -> None:
    circle_start = Circle(
        (float(start[0]), float(start[1])),
        radius=robot_radius,
        facecolor="green",
        alpha=0.6,
        edgecolor="darkgreen",
        linewidth=1.5,
        zorder=10,
    )
    ax.add_patch(circle_start)
    ax.scatter([start[0]], [start[1]], c="darkgreen", s=12, marker="o", zorder=11)

    circle_goal = Circle(
        (float(goal[0]), float(goal[1])),
        radius=robot_radius,
        facecolor="none",
        edgecolor="red",
        linewidth=1.5,
        linestyle="-",
        zorder=10,
    )
    ax.add_patch(circle_goal)
    ax.plot(goal[0], goal[1], "r*", markersize=6, zorder=11)


def style_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_nominal(ax: plt.Axes, polygon: np.ndarray, robot_radius: float) -> None:
    start = np.array([-1.0, -1.0], dtype=np.float32)
    goal = np.array([1.0, 1.0], dtype=np.float32)
    points = x_reference_straight_line(start, goal, NUM_POINTS)

    # Violation segments
    sdf = np.array([polygon_signed_distance(p, polygon) for p in points], dtype=np.float32)
    violations = sdf < robot_radius

    segments = np.stack([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=TRAJ_COLOR, linewidths=2.5, zorder=3)
    ax.add_collection(lc)
    ax.scatter(points[:, 0], points[:, 1], s=24, color=TRAJ_COLOR, zorder=4)
    if np.any(violations):
        v = points[violations]
        ax.scatter(v[:, 0], v[:, 1], s=34, color=VIOLATION_COLOR, zorder=5)

    draw_obstacle(ax, polygon, filled=True)
    draw_start_goal(ax, start, goal, robot_radius)

    style_axis(ax)


def panel_projection_geometry(ax: plt.Axes, polygon: np.ndarray, robot_radius: float) -> None:
    start = np.array([-1.0, -1.0], dtype=np.float32)
    goal = np.array([1.0, 1.0], dtype=np.float32)
    points = x_reference_straight_line(start, goal, NUM_POINTS)

    sdf, _, nominal_pt, proj, normal = compute_projection_geometry(points, polygon, robot_radius)

    # Draw local CFS inner-approximation set F(x^(k)) as
    # half-space intersection with a local trust region.
    f_poly = build_local_convex_set_polygon(
        center=nominal_pt,
        normal=normal,
        point_on_plane=proj,
        radius=0.52,
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

    draw_obstacle(ax, polygon, filled=False)
    ax.plot(points[:, 0], points[:, 1], color=TRAJ_COLOR, linewidth=2.2, zorder=2)
    ax.scatter(points[:, 0], points[:, 1], s=24, color=TRAJ_COLOR, zorder=2)
    if np.any(sdf < robot_radius):
        v = points[sdf < robot_radius]
        ax.scatter(v[:, 0], v[:, 1], s=34, color=VIOLATION_COLOR, zorder=3)
    ax.scatter([nominal_pt[0]], [nominal_pt[1]], s=70, color=VIOLATION_COLOR, zorder=4)
    ax.scatter(
        [proj[0]],
        [proj[1]],
        s=86,
        color=TRAJ_COLOR,
        edgecolors="black",
        linewidths=0.9,
        zorder=6,
    )
    ax.annotate(
        "",
        xy=proj,
        xytext=nominal_pt,
        arrowprops=dict(arrowstyle="->", color="black", linewidth=2),
        zorder=4,
    )
    tangent = np.array([-normal[1], normal[0]], dtype=np.float32)
    p1 = proj - tangent * 1.1
    p2 = proj + tangent * 1.1
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle="--", color="#999999", linewidth=1.0, zorder=1)
    # QP metric cue: three light-gray contour rings around x^(k), J(x)=c1,c2,c3.
    # Make contour size adaptive so one ring reaches projected point.
    delta = proj - nominal_pt
    dist = float(np.linalg.norm(delta))
    theta = float(np.degrees(np.arctan2(delta[1], delta[0]))) if dist > 1e-9 else 45.0
    a_mid = max(dist, 0.12)  # middle contour crosses projected point direction
    b_mid = 0.45 * a_mid
    for scale, alpha in [(0.70, 0.55), (1.00, 0.42), (1.35, 0.30)]:
        ax.add_patch(
            Ellipse(
                xy=(float(nominal_pt[0]), float(nominal_pt[1])),
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


def project_nominal_path(
    points: np.ndarray,
    polygon: np.ndarray,
    robot_radius: float,
    preferred_normal: np.ndarray,
    anchor_idx: int,
) -> np.ndarray:
    # Least-squares smoothing with obstacle constraints (projected iterations).
    corrected = points.copy()
    fallback = points.copy()
    n = len(points)
    margin = robot_radius + 0.01
    w_data = 1.0
    w_smooth = 10.0
    w_side = 0.15
    lr = 0.0035
    sigma = 2.0

    # Build a conservative fallback path (always finite / visible).
    for i in range(1, n - 1):
        sdf = polygon_signed_distance(fallback[i], polygon)
        if sdf < margin:
            proj_i, normal_i = project_to_polygon_boundary(fallback[i], polygon)
            fallback[i] = proj_i + (margin + 1e-3) * normal_i
    for _ in range(2):
        tmp = fallback.copy()
        for i in range(1, n - 1):
            tmp[i] = 0.25 * fallback[i - 1] + 0.5 * fallback[i] + 0.25 * fallback[i + 1]
        fallback = tmp
    fallback[0] = points[0]
    fallback[-1] = points[-1]

    for _ in range(220):
        grad = np.zeros_like(corrected)

        # Data term: keep close to nominal trajectory.
        grad += 2.0 * w_data * (corrected - points)

        # Smoothness term: minimize discrete curvature.
        for i in range(1, n - 1):
            d2 = corrected[i - 1] - 2.0 * corrected[i] + corrected[i + 1]
            grad[i - 1] += 2.0 * w_smooth * d2
            grad[i] += -4.0 * w_smooth * d2
            grad[i + 1] += 2.0 * w_smooth * d2

        # Side preference term: keep correction on panel-2 projected side.
        for i in range(1, n - 1):
            local_weight = math.exp(-0.5 * ((i - anchor_idx) / sigma) ** 2)
            side_val = float(np.dot(preferred_normal, corrected[i] - points[i]))
            if side_val < 0.0:
                grad[i] += -2.0 * w_side * local_weight * side_val * preferred_normal

        corrected[1:-1] -= lr * grad[1:-1]
        corrected[1:-1] = np.clip(corrected[1:-1], -1.25, 1.25)

        # Point-wise obstacle projection.
        for i in range(1, n - 1):
            sdf = polygon_signed_distance(corrected[i], polygon)
            if sdf < margin:
                proj, normal = project_to_polygon_boundary(corrected[i], polygon)
                corrected[i] = proj + (margin + 1e-3) * normal

        # Segment-wise obstacle projection (sample midpoints).
        for i in range(n - 1):
            a = corrected[i]
            b = corrected[i + 1]
            for t in (0.25, 0.5, 0.75):
                m = (1.0 - t) * a + t * b
                sdf_m = polygon_signed_distance(m, polygon)
                if sdf_m < margin:
                    proj_m, normal_m = project_to_polygon_boundary(m, polygon)
                    push = (margin - sdf_m + 1e-3) * normal_m
                    if i > 0:
                        corrected[i] += 0.5 * push
                    if i + 1 < n - 1:
                        corrected[i + 1] += 0.5 * push

        # Keep endpoints fixed.
        corrected[0] = points[0]
        corrected[-1] = points[-1]

        if not np.all(np.isfinite(corrected)):
            return fallback

    if not np.all(np.isfinite(corrected)):
        return fallback
    if float(np.max(np.abs(corrected))) > 2.0:
        return fallback

    return corrected


def panel_after_projection(ax: plt.Axes, polygon: np.ndarray, robot_radius: float) -> None:
    start = np.array([-1.0, -1.0], dtype=np.float32)
    goal = np.array([1.0, 1.0], dtype=np.float32)
    points = x_reference_straight_line(start, goal, NUM_POINTS)
    _, idx, nominal_pt, proj, normal = compute_projection_geometry(points, polygon, robot_radius)
    corrected = project_nominal_path(
        points,
        polygon,
        robot_radius,
        preferred_normal=normal,
        anchor_idx=idx,
    )
    sdf_corrected = np.array([polygon_signed_distance(p, polygon) for p in corrected], dtype=np.float32)
    violations = sdf_corrected < robot_radius

    draw_obstacle(ax, polygon, filled=True)
    ax.plot(corrected[:, 0], corrected[:, 1], color=TRAJ_COLOR, linewidth=2.5, zorder=3)
    ax.scatter(corrected[:, 0], corrected[:, 1], s=24, color=TRAJ_COLOR, zorder=4)
    if np.any(violations):
        v = corrected[violations]
        ax.scatter(v[:, 0], v[:, 1], s=34, color=VIOLATION_COLOR, zorder=5)
    # Keep panel 3 clean; projection geometry and contours are emphasized in panel 2.
    draw_start_goal(ax, start, goal, robot_radius)

    style_axis(ax)


def main() -> None:
    polygon = make_random_obstacle()
    robot_radius = 0.05

    out_dir = Path("results/cfs_panels")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panel_nominal(axes[0], polygon, robot_radius)
    panel_projection_geometry(axes[1], polygon, robot_radius)
    panel_after_projection(axes[2], polygon, robot_radius)
    plt.tight_layout()
    out_path = out_dir / "cfs_three_panels.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Also export individual panels if needed
    for idx, name in enumerate(["panel1_nominal.png", "panel2_projection.png", "panel3_after.png"]):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        if idx == 0:
            panel_nominal(ax, polygon, robot_radius)
        elif idx == 1:
            panel_projection_geometry(ax, polygon, robot_radius)
        else:
            panel_after_projection(ax, polygon, robot_radius)
        plt.tight_layout()
        fig.savefig(out_dir / name, dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"Saved: {out_dir / 'panel1_nominal.png'}")
    print(f"Saved: {out_dir / 'panel2_projection.png'}")
    print(f"Saved: {out_dir / 'panel3_after.png'}")


if __name__ == "__main__":
    main()
