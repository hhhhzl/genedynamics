"""
Stepping-stones specific metrics.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np

from genedynamics.core.types import Trajectory
from ...framework.base import MetricsPlugin


def _cvar(x: Iterable[float], alpha: float = 0.95) -> float:
    arr = np.asarray(list(x), dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    k = max(1, int(np.ceil((1.0 - float(alpha)) * arr.size)))
    tail = np.partition(arr, arr.size - k)[arr.size - k :]
    return float(np.mean(tail))


def _nearest_stone_violation(points: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=np.float32)
    d = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=-1)
    i = np.argmin(d, axis=1)
    nearest = d[np.arange(points.shape[0]), i]
    nearest_r = radii[i]
    return np.maximum(0.0, nearest - nearest_r).astype(np.float32)


class SteppingStonesMetricsPlugin(MetricsPlugin):
    @property
    def name(self) -> str:
        return "stepping_metrics"

    def compute(
        self,
        trajectory: Trajectory,
        env: Any,
        obstacles: Any,
        constraints: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = constraints
        planning_result = kwargs.get("planning_result", {}) or {}
        planning_time = float(kwargs.get("planning_time", 0.0))
        obstacle_cfg = kwargs.get("obstacle_config", {}) or {}
        success_margin = float(obstacle_cfg.get("success_margin", 0.12))
        foothold_margin = float(obstacle_cfg.get("foothold_margin", obstacle_cfg.get("robot_radius", 0.0)))

        scene = None
        if hasattr(env, "scene") and getattr(env, "scene") is not None:
            scene = getattr(env, "scene")
        elif hasattr(obstacles, "stepping_scene"):
            scene = getattr(obstacles, "stepping_scene")
        if scene is None:
            return {
                "success": False,
                "reason": "missing_scene",
                "planning_time": planning_time,
            }

        centers = np.asarray(scene.stones_centers, dtype=np.float32)
        radii = np.asarray(scene.stones_radii, dtype=np.float32)
        lmax = float(scene.l_max)
        target = np.asarray(scene.goal_mid, dtype=np.float32)

        states = np.asarray([np.asarray(s, dtype=np.float32).reshape(-1) for s in trajectory.states], dtype=np.float32)
        actions = np.asarray([np.asarray(a, dtype=np.float32).reshape(-1) for a in trajectory.actions], dtype=np.float32)

        if states.ndim != 2 or states.shape[1] < 4:
            return {
                "success": False,
                "reason": "invalid_state_shape",
                "planning_time": planning_time,
            }

        p_l = states[:, :2]
        p_r = states[:, 2:4]
        mid = 0.5 * (p_l + p_r)
        final_goal_error = float(np.linalg.norm(mid[-1] - target))

        v_l = _nearest_stone_violation(p_l, centers, radii)
        v_r = _nearest_stone_violation(p_r, centers, radii)
        foot_v = 0.5 * (v_l + v_r)

        # Step-bound should be evaluated on executed state deltas, not raw action proposals.
        step_v = np.zeros((0,), dtype=np.float32)
        if states.shape[0] >= 2:
            dl = np.linalg.norm(states[1:, :2] - states[:-1, :2], axis=1)
            dr = np.linalg.norm(states[1:, 2:4] - states[:-1, 2:4], axis=1)
            step_v = np.maximum(0.0, np.maximum(dl, dr) - lmax).astype(np.float32)

        # Keep action-step diagnostics to expose planner aggressiveness before env clipping.
        action_step_v = np.zeros((0,), dtype=np.float32)
        if actions.ndim == 2 and actions.shape[1] >= 4:
            n_l = np.linalg.norm(actions[:, :2], axis=1)
            n_r = np.linalg.norm(actions[:, 2:4], axis=1)
            action_step_v = np.maximum(0.0, np.maximum(n_l, n_r) - lmax).astype(np.float32)

        foot_v_max = float(np.max(foot_v)) if foot_v.size > 0 else 0.0
        step_v_max = float(np.max(step_v)) if step_v.size > 0 else 0.0
        success = bool(
            final_goal_error <= success_margin
            and foot_v_max <= foothold_margin
            and step_v_max <= 1e-6
        )

        cand_costs = planning_result.get("candidate_costs", None)
        n_modes = int(len(cand_costs)) if cand_costs is not None else 1
        best_idx = int(planning_result.get("best_idx", 0))

        return {
            "success": success,
            "success_goal_margin": success_margin,
            "success_foothold_margin": foothold_margin,
            "final_goal_error": final_goal_error,
            "foothold_violation_mean": float(np.mean(foot_v)) if foot_v.size > 0 else 0.0,
            "foothold_violation_max": foot_v_max,
            "foothold_violation_cvar95": _cvar(foot_v, alpha=0.95),
            "step_violation_mean": float(np.mean(step_v)) if step_v.size > 0 else 0.0,
            "step_violation_max": step_v_max,
            "step_violation_cvar95": _cvar(step_v, alpha=0.95),
            "action_step_violation_mean": float(np.mean(action_step_v)) if action_step_v.size > 0 else 0.0,
            "action_step_violation_max": float(np.max(action_step_v)) if action_step_v.size > 0 else 0.0,
            "action_step_violation_cvar95": _cvar(action_step_v, alpha=0.95),
            "planning_time": planning_time,
            "n_modes": n_modes,
            "best_idx": best_idx,
        }

