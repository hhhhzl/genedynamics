"""
Fixed D3IL avoiding obstacle generator: same cylinder layout as 2D circles.

Configured only from the experiment YAML (obstacle_config), not from DPCC projection_eval.yaml.
MBD/EBMBD can set obstacle_config.generator: d3il_avoiding_fixed.
"""

from __future__ import annotations

from typing import Dict, Any, List
import numpy as np

from enerdynamics.envs.obstacles.base import ObstacleManager
from enerdynamics.envs.obstacles.convex import SphereObstacle
from ...framework.base import ObstacleGeneratorPlugin


# Same geometry as D3IL avoiding_objects.py (6 cylinders -> 2D circles)
# mid_pos=0.5, offset=0.075, first_level_y=-0.1, level_distance=0.18
# Default: level 0 -> first 0.03, rest 0.025
D3IL_FIXED_CIRCLES = [
    {"center": np.array([0.5, -0.1], dtype=np.float32), "radius": 0.03},
    {"center": np.array([0.5 - 0.075, -0.1 + 0.18], dtype=np.float32), "radius": 0.025},
    {"center": np.array([0.5 + 0.075, -0.1 + 0.18], dtype=np.float32), "radius": 0.025},
    {"center": np.array([0.5 - 2 * 0.075, -0.1 + 2 * 0.18], dtype=np.float32), "radius": 0.025},
    {"center": np.array([0.5, -0.1 + 2 * 0.18], dtype=np.float32), "radius": 0.025},
    {"center": np.array([0.5 + 2 * 0.075, -0.1 + 2 * 0.18], dtype=np.float32), "radius": 0.025},
]

# obstacle_radius_by_level: {level: [first_radius, rest_radius]}
DEFAULT_OBSTACLE_RADIUS_BY_LEVEL = {
    0: [0.03, 0.025],
    1: [0.04, 0.035],
    2: [0.05, 0.045],
}


class D3ILAvoidingFixedGeneratorPlugin(ObstacleGeneratorPlugin):
    """
    Fixed obstacle generator for D3IL avoiding: 6 circles matching the MuJoCo cylinders.

    Config (from experiment obstacle_config, not DPCC yaml):
      - use_d3il_preset: bool (default True) — use the 6 fixed circles above.
      - fixed_cylinders: optional list of {center: [x, y], radius: r} to override.
    """

    @property
    def name(self) -> str:
        return "d3il_avoiding_fixed"

    def generate(
        self,
        level: int,
        seed: int,
        start_pos: np.ndarray,
        target_pos: np.ndarray,
        config: Dict[str, Any],
    ) -> ObstacleManager:
        _ = (seed, start_pos, target_pos)
        use_preset = config.get("use_d3il_preset", True)
        custom = config.get("fixed_cylinders", None)
        radius_by_level = config.get(
            "obstacle_radius_by_level", DEFAULT_OBSTACLE_RADIUS_BY_LEVEL
        )

        if custom is not None and len(custom) > 0:
            circles: List[Dict[str, Any]] = []
            for c in custom:
                center = np.asarray(c.get("center", [0.5, 0.0]), dtype=np.float32)
                if center.size >= 2:
                    center = center[:2]
                else:
                    center = np.array([0.5, 0.0], dtype=np.float32)
                radius = float(c.get("radius", 0.03))
                circles.append({"center": center, "radius": radius})
        elif use_preset:
            radii = radius_by_level.get(level)
            if radii is not None and len(radii) >= 2:
                r_first, r_rest = float(radii[0]), float(radii[1])
                circles = []
                for i, c in enumerate(D3IL_FIXED_CIRCLES):
                    r = r_first if i == 0 else r_rest
                    circles.append({"center": c["center"].copy(), "radius": r})
            else:
                circles = list(D3IL_FIXED_CIRCLES)
        else:
            circles = []

        obstacles = []
        for i, circ in enumerate(circles):
            # 2D circle: center (2,) and radius; SphereObstacle works in 2D for sdf/contains
            obs = SphereObstacle(
                center=circ["center"],
                radius=circ["radius"],
                name=f"d3il_fixed_{i}",
            )
            obstacles.append(obs)

        return ObstacleManager(obstacles=obstacles)
