"""
Episode outcome metrics plugin.

Provides minimal, execution-focused metrics for rollout/MPC methods:
- success (bool)
- collision (bool)

It reads these from Trajectory.info, which is populated by ExperimentRunner
when the method result dict includes `infos` / `success` / `collision`.
"""

from __future__ import annotations

from typing import Dict, Any

from genedynamics.core.types import Trajectory
from ...framework.base import MetricsPlugin


class EpisodeOutcomeMetricsPlugin(MetricsPlugin):
    @property
    def name(self) -> str:
        return "episode_outcome"

    def compute(self, trajectory: Trajectory, env: Any, obstacles: Any, constraints: Any, **kwargs: Any) -> Dict[str, Any]:
        _ = (env, obstacles, constraints, kwargs)

        info = trajectory.info or {}
        success = bool(info.get("success", False))
        collision = bool(info.get("collision", False))

        # If per-step infos exist, derive robustly
        step_infos = info.get("infos", None)
        if isinstance(step_infos, list) and step_infos:
            try:
                success = any(bool(i.get("success", False)) for i in step_infos if isinstance(i, dict)) or success
                collision = any(bool(i.get("collision", False)) for i in step_infos if isinstance(i, dict)) or collision
            except Exception:
                pass

        return {"success": bool(success), "collision": bool(collision)}


