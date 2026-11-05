"""Public T1 waypoint actor with a frozen per-mode translation scale."""

from __future__ import annotations

from typing import Any

import numpy as np

from .obstacle_feedback import OBSTACLE_BOUNDS
from .obstacle_path import PublicObstaclePathPolicy


class PublicScaledObstaclePathPolicy:
    """Shared task decoder for trained ASID and AdaptSim T1 path vectors."""

    def __init__(self, spec: Any, policy: Any, *, mode: int) -> None:
        if spec.alias != "T1" or spec.config.robot != "xarm7" or mode not in (0, 1):
            raise ValueError("scaled obstacle path is trained only for T1/xarm7")
        vector = np.asarray(policy, dtype=np.float64)
        if (vector.shape != (6,) or not np.isfinite(vector).all()
                or np.any(vector[4:] <= 0)):
            raise ValueError("scaled public path requires four waypoints and two positive scales")
        self.policy = vector.copy()
        self.base = PublicObstaclePathPolicy(spec, modes=(mode,), path=vector[:4])

    def act(self, public: dict[str, Any]) -> np.ndarray:
        action = np.asarray(self.base.act(public), dtype=np.float32)
        action[:, :3] = np.clip(
            action[:, :3] * self.policy[4 + self.base.modes][:, None],
            -OBSTACLE_BOUNDS[:3], OBSTACLE_BOUNDS[:3],
        )
        return action

    def reset(self, *, modes: Any = None) -> None:
        self.base.reset(modes=modes)

    def get_state(self) -> dict[str, Any]:
        return {"policy": self.policy.copy(), "base": self.base.get_state()}

    def restore(self, state: dict[str, Any]) -> None:
        if not np.array_equal(state["policy"], self.policy):
            raise ValueError("scaled public path differs from saved actor state")
        self.base.restore(state["base"])

    def reorder(self, indices: Any) -> None:
        self.base.reorder(indices)
