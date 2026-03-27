"""
Obstacle primitives for stepping-stones tasks.

This module defines an implicit obstacle representing the complement of
stepping-stone disks, optionally unioned with a river strip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from genedynamics.envs.obstacles.base import Obstacle, ObstacleManager
from genedynamics.tasks.stepping_stones import SteppingStonesScene


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
    Unsafe set = (outside all stones) union (inside river strip).
    """

    scene: SteppingStonesScene
    name: Optional[str] = "stepping_stones_forbidden"

    def __post_init__(self) -> None:
        self._centers = np.asarray(self.scene.stones_centers, dtype=np.float32)
        self._radii = np.asarray(self.scene.stones_radii, dtype=np.float32)
        rx0, rx1 = self.scene.river_x
        y0, y1 = self.scene.map_y
        self._river_center = np.array([0.5 * (rx0 + rx1), 0.5 * (y0 + y1)], dtype=np.float32)
        self._river_half_extents = np.array([0.5 * (rx1 - rx0), 0.5 * (y1 - y0)], dtype=np.float32)
        self.center = self._river_center
        self.bounds = (
            np.array([self.scene.map_x[0], self.scene.map_y[0]], dtype=np.float32),
            np.array([self.scene.map_x[1], self.scene.map_y[1]], dtype=np.float32),
        )

    def _safe_set_signed_margin(self, points: np.ndarray) -> np.ndarray:
        """
        Positive inside at least one stone, negative outside all stones.
        """
        diff = points[:, None, :] - self._centers[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        margins = self._radii[None, :] - dists
        return np.max(margins, axis=-1)

    def sdf(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        single = pts.ndim == 1
        if single:
            pts = pts[None, :]
        pts2 = pts[:, :2]

        # Outside-stones obstacle: negative when outside safe disks.
        safe_margin = self._safe_set_signed_margin(pts2)
        sdf_outside_stones = -safe_margin

        # River obstacle: negative inside river strip.
        sdf_river = _box_sdf(pts2, self._river_center, self._river_half_extents)

        # Union of unsafe regions.
        sdf_union = np.minimum(sdf_outside_stones, sdf_river)
        return float(sdf_union[0]) if single else sdf_union

    def contains(self, point: np.ndarray) -> bool:
        return bool(float(self.sdf(np.asarray(point, dtype=np.float32)[:2])) < 0.0)

    def distance(self, point: np.ndarray) -> float:
        return max(0.0, float(self.sdf(np.asarray(point, dtype=np.float32)[:2])))

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

