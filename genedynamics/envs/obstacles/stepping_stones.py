"""
Obstacle primitives for stepping-stones tasks.

This module defines an implicit obstacle representing the complement of
stepping-stone disks, optionally unioned with a river strip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
try:
    import jax
    import jax.numpy as jnp
except Exception:
    jax = None
    jnp = None

from genedynamics.envs.obstacles.base import Obstacle, ObstacleManager
from genedynamics.tasks.stepping_stones import SteppingStonesScene


def foot_stepping_violation_np(
    p: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    platforms: np.ndarray,
    stone_margin: float,
) -> float:
    """
    Scalar violation: 0 if foot lies on a safe disk or support platform (with margin), else positive distance to safe set.
    """
    p = np.asarray(p, dtype=np.float32).reshape(2)
    m = float(stone_margin)
    if centers is not None and np.asarray(centers).size > 0 and np.asarray(centers).shape[0] > 0:
        cc = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
        rr = np.asarray(radii, dtype=np.float32).reshape(-1)
        d = np.linalg.norm(cc - p[None, :], axis=-1)
        v_disk = float(np.min(np.maximum(d - (rr - m), 0.0)))
    else:
        v_disk = float("inf")
    plat = np.asarray(platforms, dtype=np.float32).reshape(-1, 4) if platforms is not None else np.zeros((0, 4), dtype=np.float32)
    if plat.shape[0] == 0:
        v_plat = float("inf")
    else:
        v_plat = float("inf")
        for k in range(plat.shape[0]):
            xmin, xmax, ymin, ymax = float(plat[k, 0]), float(plat[k, 1]), float(plat[k, 2]), float(plat[k, 3])
            xl, xr = xmin + m, xmax - m
            yb, yt = ymin + m, ymax - m
            if xr <= xl or yt <= yb:
                continue
            if xl <= float(p[0]) <= xr and yb <= float(p[1]) <= yt:
                v_plat = 0.0
                break
            dx = max(0.0, xl - float(p[0])) + max(0.0, float(p[0]) - xr)
            dy = max(0.0, yb - float(p[1])) + max(0.0, float(p[1]) - yt)
            v_plat = min(v_plat, float(np.hypot(dx, dy)))
    return float(min(v_disk, v_plat))


def _platform_union_margin_np(points: np.ndarray, platforms: np.ndarray) -> np.ndarray:
    """Max over platforms of min edge distance (positive inside axis-aligned rectangle)."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if platforms is None or np.asarray(platforms).size == 0:
        return np.full((pts.shape[0],), -np.inf, dtype=np.float32)
    plat = np.asarray(platforms, dtype=np.float32).reshape(-1, 4)
    best = np.full((pts.shape[0],), -np.inf, dtype=np.float32)
    for k in range(plat.shape[0]):
        xmin, xmax, ymin, ymax = float(plat[k, 0]), float(plat[k, 1]), float(plat[k, 2]), float(plat[k, 3])
        m = np.minimum.reduce(
            [pts[:, 0] - xmin, xmax - pts[:, 0], pts[:, 1] - ymin, ymax - pts[:, 1]]
        ).astype(np.float32)
        best = np.maximum(best, m)
    return best


def _box_sdf(points: np.ndarray, center: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    """
    Signed distance to axis-aligned rectangle (negative inside).
    """
    q = np.abs(points - center[None, :]) - half_extents[None, :]
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


@dataclass
class SteppingStonesForbiddenRegionObstacle(Obstacle):
    """
    Unsafe set = outside all stones, optionally unioned with the river strip.
    """

    scene: SteppingStonesScene
    name: Optional[str] = "stepping_stones_forbidden"

    def __post_init__(self) -> None:
        self._centers = np.asarray(self.scene.stones_centers, dtype=np.float32)
        self._radii = np.asarray(self.scene.stones_radii, dtype=np.float32)
        raw_plat = np.asarray(getattr(self.scene, "support_platforms", np.zeros((0, 4))), dtype=np.float32)
        self._platforms = raw_plat
        self._n_platforms = int(raw_plat.shape[0])
        self._platforms_pad = np.zeros((4, 4), dtype=np.float32)
        if self._n_platforms > 0:
            self._platforms_pad[: self._n_platforms] = raw_plat[:4]
        rx0, rx1 = self.scene.river_x
        y0, y1 = self.scene.map_y
        self._has_river = bool(getattr(self.scene, "has_river", True))
        self._river_center = np.array([0.5 * (rx0 + rx1), 0.5 * (y0 + y1)], dtype=np.float32)
        self._river_half_extents = np.array([0.5 * (rx1 - rx0), 0.5 * (y1 - y0)], dtype=np.float32)
        self.center = self._river_center
        self.bounds = (
            np.array([self.scene.map_x[0], self.scene.map_y[0]], dtype=np.float32),
            np.array([self.scene.map_x[1], self.scene.map_y[1]], dtype=np.float32),
        )

    def _safe_set_signed_margin(self, points: np.ndarray) -> np.ndarray:
        """
        Positive inside at least one stone or start/goal platform, negative otherwise.
        """
        if self._centers.shape[0] > 0:
            diff = points[:, None, :] - self._centers[None, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            margins = self._radii[None, :] - dists
            disk_m = np.max(margins, axis=-1)
        else:
            disk_m = np.full((points.shape[0],), -np.inf, dtype=np.float32)
        plat_m = _platform_union_margin_np(points, self._platforms)
        return np.maximum(disk_m, plat_m)

    def sdf(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        single = pts.ndim == 1
        if single:
            pts = pts[None, :]
        pts2 = pts[:, :2]

        # High ``sdf`` = safer. Feasible for CFS when ``sdf >= clearance`` (i.e. ``g=clearance-sdf<=0``).
        # ``contains`` / ``distance`` treat ``sdf<0`` as inside the forbidden union (off stones / in river).
        safe_margin = self._safe_set_signed_margin(pts2)
        sdf_outside_stones = safe_margin

        if self._has_river:
            sdf_river = _box_sdf(pts2, self._river_center, self._river_half_extents)
            sdf_union = np.minimum(sdf_outside_stones, sdf_river)
        else:
            sdf_union = sdf_outside_stones
        return float(sdf_union[0]) if single else sdf_union

    def jax_sdf(self, points):
        if jnp is None:
            raise RuntimeError("JAX required for jax_sdf")
        pts = jnp.asarray(points, dtype=jnp.float32)
        single = pts.ndim == 1
        if single:
            pts = pts[None, :]
        # EB-MBD / batched solvers may pass (B, M, H, 2) or similar; never use pts[:, :2] on rank>2.
        lead_shape = None
        if pts.ndim > 2:
            lead_shape = pts.shape[:-1]
            pts = pts.reshape(-1, int(pts.shape[-1]))
        elif pts.ndim == 1:
            pts = pts.reshape(1, -1)
        pts2 = pts[:, :2]
        centers = jnp.asarray(self._centers, dtype=jnp.float32)
        radii = jnp.asarray(self._radii, dtype=jnp.float32)
        plat_pad = jnp.asarray(self._platforms_pad, dtype=jnp.float32)
        n_plat = int(self._n_platforms)

        def _plat_margin_batch(p2):
            def one_bounds(bounds):
                xm, xM, ym, yM = bounds[0], bounds[1], bounds[2], bounds[3]
                return jnp.minimum(
                    jnp.minimum(p2[:, 0] - xm, xM - p2[:, 0]),
                    jnp.minimum(p2[:, 1] - ym, yM - p2[:, 1]),
                )

            m0 = one_bounds(plat_pad[0])
            m1 = one_bounds(plat_pad[1])
            m2 = one_bounds(plat_pad[2])
            m3 = one_bounds(plat_pad[3])
            stacked = jnp.stack([m0, m1, m2, m3], axis=1)
            mask = jnp.array(
                [1.0 if k < n_plat else 0.0 for k in range(4)],
                dtype=jnp.float32,
            )
            masked = jnp.where(mask[None, :] > 0.5, stacked, -1e9)
            return jnp.max(masked, axis=1)

        if centers.shape[0] > 0:
            diff = pts2[:, None, :] - centers[None, :, :]
            dists = jnp.linalg.norm(diff, axis=-1)
            disk_m = jnp.max(radii[None, :] - dists, axis=-1)
        else:
            disk_m = jnp.full((pts2.shape[0],), -jnp.inf, dtype=jnp.float32)
        plat_vals = _plat_margin_batch(pts2)
        safe_margin = jnp.maximum(disk_m, plat_vals)
        sdf_outside_stones = safe_margin
        if self._has_river:
            center = jnp.asarray(self._river_center, dtype=jnp.float32)
            half = jnp.asarray(self._river_half_extents, dtype=jnp.float32)
            q = jnp.abs(pts2 - center[None, :]) - half[None, :]
            outside = jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1)
            inside = jnp.minimum(jnp.max(q, axis=-1), 0.0)
            sdf_river = outside + inside
            sdf_union = jnp.minimum(sdf_outside_stones, sdf_river)
        else:
            sdf_union = sdf_outside_stones
        if lead_shape is not None:
            sdf_union = sdf_union.reshape(lead_shape)
        return sdf_union[0] if single else sdf_union

    def jax_gradient(self, point):
        if jax is None or jnp is None:
            raise RuntimeError("JAX required for jax_gradient")
        p = jnp.asarray(point, dtype=jnp.float32).reshape(-1)[:2]
        grad_fn = jax.grad(lambda x: self.jax_sdf(x))
        return grad_fn(p)

    def contains(self, point: np.ndarray) -> bool:
        return bool(float(self.sdf(np.asarray(point, dtype=np.float32)[:2])) < 0.0)

    def distance(self, point: np.ndarray) -> float:
        return max(0.0, -float(self.sdf(np.asarray(point, dtype=np.float32)[:2])))

    def gradient(self, point: np.ndarray) -> np.ndarray:
        p = np.asarray(point, dtype=np.float32).reshape(-1)[:2]
        eps = 1e-3
        ex = np.array([eps, 0.0], dtype=np.float32)
        ey = np.array([0.0, eps], dtype=np.float32)
        gx = (float(self.sdf(p + ex)) - float(self.sdf(p - ex))) / (2.0 * eps)
        gy = (float(self.sdf(p + ey)) - float(self.sdf(p - ey))) / (2.0 * eps)
        return np.array([gx, gy], dtype=np.float32)

    def to_backend(self, backend: str):
        return {
            "type": "stepping_stones_forbidden",
            "backend": backend,
            "scene_level": int(self.scene.level),
            "difficulty": self.scene.difficulty,
        }


def make_stepping_stones_obstacles(scene: SteppingStonesScene) -> ObstacleManager:
    manager = ObstacleManager()
    manager.add(SteppingStonesForbiddenRegionObstacle(scene=scene))
    # Attach scene metadata for env/metrics/plugins without adding ad-hoc side channels.
    manager.stepping_scene = scene
    return manager

