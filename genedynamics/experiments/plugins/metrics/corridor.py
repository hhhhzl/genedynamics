"""
Corridor obstacle avoidance metrics for humanoid planning.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from genedynamics.core.types import Trajectory
from ...framework.base import MetricsPlugin


def _cvar(x: np.ndarray, alpha: float = 0.95) -> float:
    arr = np.asarray(x, dtype=np.float32).ravel()
    if arr.size == 0:
        return 0.0
    k = max(1, int(np.ceil((1.0 - alpha) * arr.size)))
    tail = np.partition(arr, arr.size - k)[arr.size - k:]
    return float(np.mean(tail))


class CorridorMetricsPlugin(MetricsPlugin):
    @property
    def name(self) -> str:
        return "corridor_metrics"

    def compute(
        self,
        trajectory: Trajectory,
        env: Any,
        obstacles: Any,
        constraints: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        planning_result = kwargs.get("planning_result", {}) or {}
        planning_time = float(kwargs.get("planning_time", 0.0))
        obstacle_cfg = kwargs.get("obstacle_config", {}) or {}
        success_margin = float(obstacle_cfg.get("success_margin", 0.20))

        states = np.asarray(
            [np.asarray(s, dtype=np.float32).ravel() for s in trajectory.states],
            dtype=np.float32,
        )
        actions = np.asarray(
            [np.asarray(a, dtype=np.float32).ravel() for a in trajectory.actions],
            dtype=np.float32,
        )

        if states.ndim != 2 or states.shape[1] < 14:
            return {
                "success": False,
                "reason": "invalid_state_shape",
                "planning_time": planning_time,
            }

        target = np.asarray(env.target, dtype=np.float32)
        T = states.shape[0]

        # Basic position metrics.
        final_pos = states[-1, :2]
        final_goal_error = float(np.linalg.norm(final_pos - target))

        # Collision metrics.
        clearances = np.array(
            [env.get_min_clearance(states[t]) for t in range(T)],
            dtype=np.float32,
        )
        min_clearance = float(np.min(clearances))
        collision_count = int(np.sum(clearances < 0.0))
        collision_steps = [int(t) for t in range(T) if clearances[t] < 0.0]

        # Path metrics.
        if T >= 2:
            diffs = np.linalg.norm(states[1:, :2] - states[:-1, :2], axis=1)
            path_length = float(np.sum(diffs))
            straight_dist = float(np.linalg.norm(target - states[0, :2]))
            path_efficiency = straight_dist / max(path_length, 1e-6)
        else:
            path_length = 0.0
            path_efficiency = 0.0

        # Control smoothness.
        if actions.shape[0] >= 2:
            du = np.linalg.norm(actions[1:] - actions[:-1], axis=1)
            smoothness = float(np.mean(du))
        else:
            smoothness = 0.0

        # Posture usage metrics.
        max_crouch = float(np.min(states[:, 3]))   # min height = max crouch
        max_torso_yaw = float(np.max(np.abs(states[:, 4])))
        max_arm_tuck_L = float(np.max(states[:, 5]))
        max_arm_tuck_R = float(np.max(states[:, 6]))

        # Progress metric.
        x_progress = float(states[-1, 0] - states[0, 0])
        total_x = float(target[0] - states[0, 0])
        progress_ratio = x_progress / max(total_x, 1e-6)

        # Time to reach goal (first step within margin).
        completion_step = T
        for t in range(T):
            if np.linalg.norm(states[t, :2] - target) < success_margin:
                completion_step = t
                break

        # Success criteria.
        success = bool(
            final_goal_error <= success_margin
            and collision_count == 0
        )

        cand_costs = planning_result.get("candidate_costs", None)
        n_modes = int(len(cand_costs)) if cand_costs is not None else 1
        best_idx = int(planning_result.get("best_idx", 0))

        return {
            "success": success,
            "final_goal_error": final_goal_error,
            "collision_count": collision_count,
            "collision_steps": collision_steps,
            "min_clearance": min_clearance,
            "clearance_mean": float(np.mean(clearances)),
            "clearance_cvar95": _cvar(-clearances, alpha=0.95),
            "path_length": path_length,
            "path_efficiency": path_efficiency,
            "smoothness": smoothness,
            "max_crouch": max_crouch,
            "max_torso_yaw_deg": float(np.degrees(max_torso_yaw)),
            "max_arm_tuck_L": max_arm_tuck_L,
            "max_arm_tuck_R": max_arm_tuck_R,
            "x_progress": x_progress,
            "progress_ratio": progress_ratio,
            "completion_step": completion_step,
            "planning_time": planning_time,
            "n_modes": n_modes,
            "best_idx": best_idx,
        }
