"""
Corridor 2D obstacle generator plugin.

Creates lightweight Obstacle wrappers around each CorridorObstacle in the
environment's scene.  All SDF / gradient methods use **jnp** operations so
that JAX can trace and differentiate through them — this is critical for
the CFS QP JAX path to build correct constraint gradients.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None

from genedynamics.envs.obstacles.base import ObstacleManager
from ...framework.base import ObstacleGeneratorPlugin


def _box_sdf_jax(pt, xmin, xmax, ymin, ymax):
    """AABB signed distance — pure jnp, JAX-traceable."""
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    hx = 0.5 * (xmax - xmin)
    hy = 0.5 * (ymax - ymin)
    dx = jnp.abs(pt[0] - cx) - hx
    dy = jnp.abs(pt[1] - cy) - hy
    outside = jnp.sqrt(jnp.maximum(dx, 0.0) ** 2 + jnp.maximum(dy, 0.0) ** 2 + 1e-12)
    inside = jnp.minimum(jnp.maximum(dx, dy), 0.0)
    return outside + inside


def _sphere_sdf_jax(pt, cx, cy, r):
    """Circle signed distance — pure jnp, JAX-traceable."""
    return jnp.sqrt((pt[0] - cx) ** 2 + (pt[1] - cy) ** 2 + 1e-12) - r


def _wall_sdf_jax(pt, y_boundary, direction):
    """Half-plane wall SDF — pure jnp, JAX-traceable."""
    return direction * (y_boundary - pt[1])


class _CorridorBoxObstacle:
    """Lightweight Obstacle wrapper for an axis-aligned box obstacle.
    All SDF/gradient ops use jnp for JAX traceability."""

    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float, name: str = ""):
        self._xmin = float(x_min)
        self._xmax = float(x_max)
        self._ymin = float(y_min)
        self._ymax = float(y_max)
        self.center = np.array([0.5 * (x_min + x_max), 0.5 * (y_min + y_max)], dtype=np.float32)
        self.bounds = (
            np.array([x_min, y_min], dtype=np.float32),
            np.array([x_max, y_max], dtype=np.float32),
        )
        self.name = name

    def sdf(self, points):
        """NumPy-compatible SDF (returns float for scalar input, ndarray for batch)."""
        p = np.asarray(points, dtype=np.float32)
        single = p.ndim == 1
        if single:
            p = p.reshape(1, -1)
        cx = 0.5 * (self._xmin + self._xmax)
        cy = 0.5 * (self._ymin + self._ymax)
        hx = 0.5 * (self._xmax - self._xmin)
        hy = 0.5 * (self._ymax - self._ymin)
        dx = np.abs(p[:, 0] - cx) - hx
        dy = np.abs(p[:, 1] - cy) - hy
        outside = np.sqrt(np.maximum(dx, 0.0) ** 2 + np.maximum(dy, 0.0) ** 2 + 1e-12)
        inside = np.minimum(np.maximum(dx, dy), 0.0)
        result = outside + inside
        return float(result[0]) if single else result

    def jax_sdf(self, points):
        """JAX-traceable SDF (required by CFS QP JAX path)."""
        pt = jnp.asarray(points, dtype=jnp.float32).reshape(-1)[:2]
        return _box_sdf_jax(pt, self._xmin, self._xmax, self._ymin, self._ymax)

    def jax_gradient(self, points):
        """JAX-traceable gradient (required by CFS QP JAX path)."""
        import jax
        pt = jnp.asarray(points, dtype=jnp.float32).reshape(-1)[:2]
        g = jax.grad(lambda p: _box_sdf_jax(p, self._xmin, self._xmax, self._ymin, self._ymax))(pt)
        return jnp.where(jnp.isnan(g), 0.0, g)

    def contains(self, point) -> bool:
        p = np.asarray(point, dtype=np.float32).ravel()
        return bool(self._xmin <= p[0] <= self._xmax and self._ymin <= p[1] <= self._ymax)

    def distance(self, point):
        return jnp.maximum(0.0, self.sdf(point))

    def gradient(self, point):
        import jax
        pt = jnp.asarray(point, dtype=jnp.float32).reshape(-1)[:2]
        g = jax.grad(lambda p: _box_sdf_jax(p, self._xmin, self._xmax, self._ymin, self._ymax))(pt)
        return jnp.where(jnp.isnan(g), 0.0, g)

    def to_backend(self, backend: str) -> Any:
        return self


class _CorridorSphereObstacle:
    """Lightweight Obstacle wrapper for a spherical obstacle.
    All SDF/gradient ops use jnp for JAX traceability."""

    def __init__(self, cx: float, cy: float, radius: float, name: str = ""):
        self._cx = float(cx)
        self._cy = float(cy)
        self._r = float(radius)
        self.center = np.array([cx, cy], dtype=np.float32)
        self.bounds = (
            np.array([cx - radius, cy - radius], dtype=np.float32),
            np.array([cx + radius, cy + radius], dtype=np.float32),
        )
        self.name = name

    def sdf(self, points):
        p = np.asarray(points, dtype=np.float32)
        single = p.ndim == 1
        if single:
            p = p.reshape(1, -1)
        d = np.sqrt((p[:, 0] - self._cx) ** 2 + (p[:, 1] - self._cy) ** 2 + 1e-12) - self._r
        return float(d[0]) if single else d

    def jax_sdf(self, points):
        pt = jnp.asarray(points, dtype=jnp.float32).reshape(-1)[:2]
        return _sphere_sdf_jax(pt, self._cx, self._cy, self._r)

    def jax_gradient(self, points):
        import jax
        pt = jnp.asarray(points, dtype=jnp.float32).reshape(-1)[:2]
        g = jax.grad(lambda p: _sphere_sdf_jax(p, self._cx, self._cy, self._r))(pt)
        return jnp.where(jnp.isnan(g), 0.0, g)

    def contains(self, point) -> bool:
        p = np.asarray(point, dtype=np.float32).ravel()
        return bool(np.sqrt((p[0] - self._cx) ** 2 + (p[1] - self._cy) ** 2) <= self._r)

    def distance(self, point):
        return jnp.maximum(0.0, self.sdf(point))

    def gradient(self, point):
        return self.jax_gradient(point)

    def to_backend(self, backend: str) -> Any:
        return self


class _CorridorWallObstacle:
    """Wall represented as a half-plane obstacle.
    All SDF/gradient ops use jnp for JAX traceability."""

    def __init__(self, y_boundary: float, direction: float, length: float, name: str = ""):
        self._y = float(y_boundary)
        self._dir = float(direction)
        self._length = float(length)
        self.center = np.array([length / 2, y_boundary], dtype=np.float32)
        self.bounds = (
            np.array([0.0, y_boundary - 0.1], dtype=np.float32),
            np.array([length, y_boundary + 0.1], dtype=np.float32),
        )
        self.name = name

    def sdf(self, points):
        p = np.asarray(points, dtype=np.float32)
        single = p.ndim == 1
        if single:
            p = p.reshape(1, -1)
        d = self._dir * (self._y - p[:, 1])
        return float(d[0]) if single else d

    def jax_sdf(self, points):
        pt = jnp.asarray(points, dtype=jnp.float32).reshape(-1)[:2]
        return _wall_sdf_jax(pt, self._y, self._dir)

    def jax_gradient(self, points):
        return jnp.array([0.0, -self._dir], dtype=jnp.float32)

    def contains(self, point) -> bool:
        p = np.asarray(point, dtype=np.float32).ravel()
        return bool(self._dir * (p[1] - self._y) > 0)

    def distance(self, point):
        return jnp.maximum(0.0, self.sdf(point))

    def gradient(self, point):
        return self.jax_gradient(point)

    def to_backend(self, backend: str) -> Any:
        return self


class Corridor2DObstacleGeneratorPlugin(ObstacleGeneratorPlugin):
    @property
    def name(self) -> str:
        return "corridor_2d"

    def generate(
        self,
        level: int,
        seed: int,
        start_pos: np.ndarray,
        target_pos: np.ndarray,
        config: Dict[str, Any],
    ) -> ObstacleManager:
        from genedynamics.envs.domains.humanoid.corridor import HumanoidCorridor2DEnv
        preset = config.get("scene_preset", "medium")
        env = HumanoidCorridor2DEnv(scene_preset=preset)
        scene = env.scene

        obs_list = []
        # Corridor walls (always full-height, no z-gating needed).
        obs_list.append(_CorridorWallObstacle(
            scene.wall_y_max, 1.0, scene.corridor_length, "wall_top"))
        obs_list.append(_CorridorWallObstacle(
            scene.wall_y_min, -1.0, scene.corridor_length, "wall_bottom"))

        # Only full-height obstacles in the ObstacleManager.
        for co in scene.obstacles:
            is_full_height = co.z_min <= 0.05 and co.z_max >= 1.8
            if not is_full_height:
                continue
            if getattr(co, "shape", "box") == "sphere":
                obs_list.append(_CorridorSphereObstacle(
                    co.cx, co.cy, co.radius, co.name))
            else:
                obs_list.append(_CorridorBoxObstacle(
                    co.x_min, co.x_max, co.y_min, co.y_max, co.name))

        return ObstacleManager(obstacles=obs_list)
