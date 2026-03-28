"""
Stepping-stones 2D scene generator plugin.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from genedynamics.envs.obstacles.base import ObstacleManager
from genedynamics.envs.obstacles.stepping_stones import make_stepping_stones_obstacles
from genedynamics.tasks.stepping_stones import sample_stepping_stones_scene
from ...framework.base import ObstacleGeneratorPlugin


class SteppingStones2DObstacleGeneratorPlugin(ObstacleGeneratorPlugin):
    @property
    def name(self) -> str:
        return "stepping_stones_2d"

    def generate(
        self,
        level: int,
        seed: int,
        start_pos: np.ndarray,
        target_pos: np.ndarray,
        config: Dict[str, Any],
    ) -> ObstacleManager:
        s = np.asarray(start_pos, dtype=np.float32).reshape(-1)
        t = np.asarray(target_pos, dtype=np.float32).reshape(-1)
        if s.size >= 4:
            start_mid = (0.5 * (s[:2] + s[2:4])).tolist()
        else:
            start_mid = (s[:2] if s.size >= 2 else np.array([-1.25, 0.0], dtype=np.float32)).tolist()
        if t.size >= 4:
            goal_mid = (0.5 * (t[:2] + t[2:4])).tolist()
        else:
            goal_mid = (t[:2] if t.size >= 2 else np.array([1.25, 0.0], dtype=np.float32)).tolist()

        scene = sample_stepping_stones_scene(
            level=max(1, int(level)),
            seed=int(seed),
            l_max=float(config.get("l_max", 0.35)),
            stance_width=float(config.get("stance_width", 0.30)),
            start_mid=(float(start_mid[0]), float(start_mid[1])),
            goal_mid=(float(goal_mid[0]), float(goal_mid[1])),
        )
        return make_stepping_stones_obstacles(scene)

