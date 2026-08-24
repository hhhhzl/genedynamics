"""No-obstacle generator for task-owned contact scenes."""

from typing import Any, Dict

import numpy as np

from genedynamics.envs.obstacles.base import ObstacleManager
from ...framework.base import ObstacleGeneratorPlugin


class EmptyObstacleGeneratorPlugin(ObstacleGeneratorPlugin):
    """Return an empty manager without injecting geometry into the task model."""

    @property
    def name(self) -> str:
        return "none"

    def generate(
        self,
        level: Any,
        seed: int,
        start_pos: Any,
        target_pos: np.ndarray,
        config: Dict[str, Any],
    ) -> ObstacleManager:
        del level, seed, start_pos, target_pos, config
        return ObstacleManager()


__all__ = ["EmptyObstacleGeneratorPlugin"]
