"""
Stepping-stones specific metrics.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np

from genedynamics.core.types import Trajectory
from genedynamics.tasks.stepping_stones import decode_plan_states
from ...framework.base import MetricsPlugin


def _cvar(x: Iterable[float], alpha: float = 0.95) -> float:
    arr = np.asarray(list(x), dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    k = max(1, int(np.ceil((1.0 - float(alpha)) * arr.size)))
    tail = np.partition(arr, arr.size - k)[arr.size - k :]
    return float(np.mean(tail))


def _swing_and_stance(mode: int) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    m = int(mode) % 4
    if m == 0:
        return ("FR", "RL"), ("FL", "RR")
    if m == 2:
        return ("FL", "RR"), ("FR", "RL")
    return tuple(), ("FL", "FR", "RL", "RR")


def _support_violation(
    points: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    river_x: np.ndarray,
    *,
    has_river: bool,
) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32)
    out = np.zeros((pts.shape[0],), dtype=np.float32)
    for i, p in enumerate(pts):
        d = np.linalg.norm(centers - p[None, :], axis=-1)
        stone_v = float(np.min(np.maximum(d - radii, 0.0))) if centers.size > 0 else 1.0
        if has_river:
            if float(p[0]) <= float(river_x[0]) or float(p[0]) >= float(river_x[1]):
                out[i] = 0.0
                continue
            bank_v = min(abs(float(p[0]) - float(river_x[0])), abs(float(p[0]) - float(river_x[1])))
            out[i] = float(min(stone_v, bank_v))
        else:
            out[i] = stone_v
    return out


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
        river_x = np.asarray(scene.river_x, dtype=np.float32)
        has_river = bool(getattr(scene, "has_river", True))
        lmax = float(scene.l_max)
        target = np.asarray(scene.goal_mid, dtype=np.float32)
        body_step_limit = float(getattr(env, "body_shift_limit", lmax))
        swing_step_limit = float(getattr(env, "swing_step_limit", lmax))

        states = np.asarray([np.asarray(s, dtype=np.float32).reshape(-1) for s in trajectory.states], dtype=np.float32)
        actions = np.asarray([np.asarray(a, dtype=np.float32).reshape(-1) for a in trajectory.actions], dtype=np.float32)

        if states.ndim != 2 or states.shape[1] < 3:
            return {
                "success": False,
                "reason": "invalid_state_shape",
                "planning_time": planning_time,
            }

        body, yaw, feet, mode = decode_plan_states(
            states,
            step_width=float(getattr(env, "step_width", getattr(env, "stance_width", 0.30))),
            half_pair_length=float(getattr(env, "fore_hind_offset", 0.18)),
        )
        goal_feet = env._nominal_feet(target, float(yaw[-1]))
        final_goal_error = float(np.linalg.norm(body[-1] - target))
        terminal_foot_error = np.asarray(
            [np.linalg.norm(feet[leg][-1] - goal_feet[leg]) for leg in ("FL", "FR", "RL", "RR")],
            dtype=np.float32,
        )

        foot_v_all = []
        for leg in ("FL", "FR", "RL", "RR"):
            foot_v_all.append(_support_violation(feet[leg], centers, radii, river_x, has_river=has_river))
        foot_v = np.mean(np.stack(foot_v_all, axis=0), axis=0).astype(np.float32)

        step_v = np.zeros((0,), dtype=np.float32)
        stance_drift = np.zeros((0,), dtype=np.float32)
        if states.shape[0] >= 2:
            body_step = np.linalg.norm(body[1:] - body[:-1], axis=1)
            swing_step = []
            drift = []
            for t in range(states.shape[0] - 1):
                swing_legs, stance_legs = _swing_and_stance(int(mode[t]))
                if swing_legs:
                    swing_step.append(
                        max(np.linalg.norm(feet[leg][t + 1] - feet[leg][t]) for leg in swing_legs)
                    )
                else:
                    swing_step.append(0.0)
                drift.append(
                    max(np.linalg.norm(feet[leg][t + 1] - feet[leg][t]) for leg in stance_legs)
                )
            step_v = np.maximum(0.0, np.maximum(body_step - body_step_limit, np.asarray(swing_step) - swing_step_limit)).astype(np.float32)
            stance_drift = np.asarray(drift, dtype=np.float32)

        action_step_v = np.zeros((0,), dtype=np.float32)
        if actions.ndim == 2 and actions.shape[1] >= 6:
            body_a = np.linalg.norm(actions[:, :2], axis=1)
            swing_a = np.linalg.norm(actions[:, 3:5], axis=1)
            phase_a = np.maximum(0.0, np.maximum(actions[:, 5] - float(getattr(env, "phase_rate_limit", 1.0)), -actions[:, 5]))
            action_step_v = np.maximum(
                0.0,
                np.maximum(np.maximum(body_a - float(getattr(env, "body_acc_limit", body_step_limit)), swing_a - float(getattr(env, "length_rate_limit", swing_step_limit))), phase_a),
            ).astype(np.float32)

        foot_v_max = float(np.max(foot_v)) if foot_v.size > 0 else 0.0
        step_v_max = float(np.max(step_v)) if step_v.size > 0 else 0.0
        terminal_foot_error_max = float(np.max(terminal_foot_error)) if terminal_foot_error.size > 0 else 0.0
        terminal_foot_error_mean = float(np.mean(terminal_foot_error)) if terminal_foot_error.size > 0 else 0.0
        success = bool(
            final_goal_error <= success_margin
            and foot_v_max <= foothold_margin
            and step_v_max <= 1e-6
            and terminal_foot_error_max <= float(getattr(env, "terminal_foot_margin", 0.20))
        )

        cand_costs = planning_result.get("candidate_costs", None)
        n_modes = int(len(cand_costs)) if cand_costs is not None else 1
        best_idx = int(planning_result.get("best_idx", 0))

        return {
            "success": success,
            "success_goal_margin": success_margin,
            "success_foothold_margin": foothold_margin,
            "final_goal_error": final_goal_error,
            "terminal_foot_error_mean": terminal_foot_error_mean,
            "terminal_foot_error_max": terminal_foot_error_max,
            "foothold_violation_mean": float(np.mean(foot_v)) if foot_v.size > 0 else 0.0,
            "foothold_violation_max": foot_v_max,
            "foothold_violation_cvar95": _cvar(foot_v, alpha=0.95),
            "step_violation_mean": float(np.mean(step_v)) if step_v.size > 0 else 0.0,
            "step_violation_max": step_v_max,
            "step_violation_cvar95": _cvar(step_v, alpha=0.95),
            "stance_drift_mean": float(np.mean(stance_drift)) if stance_drift.size > 0 else 0.0,
            "stance_drift_max": float(np.max(stance_drift)) if stance_drift.size > 0 else 0.0,
            "action_step_violation_mean": float(np.mean(action_step_v)) if action_step_v.size > 0 else 0.0,
            "action_step_violation_max": float(np.max(action_step_v)) if action_step_v.size > 0 else 0.0,
            "action_step_violation_cvar95": _cvar(action_step_v, alpha=0.95),
            "planning_time": planning_time,
            "n_modes": n_modes,
            "best_idx": best_idx,
        }

