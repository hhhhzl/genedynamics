"""
Stepping-stones specific metrics.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from genedynamics.core.types import Trajectory
from genedynamics.envs.obstacles.stepping_stones import foot_stepping_violation_np
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
    if int(mode) % 2 == 0:
        return ("FR", "RL"), ("FL", "RR")
    return ("FL", "RR"), ("FR", "RL")


def _support_violation(
    points: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    river_x: np.ndarray,
    *,
    has_river: bool,
    stone_margin: float = 0.0,
    platforms: Optional[np.ndarray] = None,
) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32)
    plat = platforms if platforms is not None else np.zeros((0, 4), dtype=np.float32)
    out = np.zeros((pts.shape[0],), dtype=np.float32)
    for i, p in enumerate(pts):
        stone_v = foot_stepping_violation_np(p, centers, radii, plat, float(stone_margin))
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

    @staticmethod
    def _follower_metrics(
        body: np.ndarray, yaw: np.ndarray,
        feet: Dict[str, np.ndarray], mode: np.ndarray,
        l_max: float,
    ) -> Dict[str, float]:
        """Metrics that predict how easy the trajectory is for a PD follower."""
        T = body.shape[0]
        out: Dict[str, float] = {}
        # 1. Body velocity variance: stop-and-go vs smooth forward progress
        if T >= 2:
            body_vel = np.linalg.norm(body[1:] - body[:-1], axis=1)
            out["body_velocity_mean"] = float(np.mean(body_vel))
            out["body_velocity_var"] = float(np.var(body_vel))
        else:
            out["body_velocity_mean"] = 0.0
            out["body_velocity_var"] = 0.0
        # 2. Max foot reach ratio: foot step / l_max (1.0 = at kinematic limit)
        max_reach = 0.0
        if T >= 2:
            for leg in ("FL", "FR", "RL", "RR"):
                steps = np.linalg.norm(feet[leg][1:] - feet[leg][:-1], axis=1)
                if steps.size > 0:
                    max_reach = max(max_reach, float(np.max(steps)))
        out["foot_reach_ratio_max"] = max_reach / max(l_max, 1e-6)
        # 3. Yaw rate max
        if T >= 2:
            dyaw = np.abs(yaw[1:] - yaw[:-1])
            out["yaw_rate_max"] = float(np.max(dyaw))
        else:
            out["yaw_rate_max"] = 0.0
        # 4. Max lateral body deviation from start→goal line
        if T >= 2:
            direction = body[-1] - body[0]
            d_norm = np.linalg.norm(direction)
            if d_norm > 1e-6:
                d_hat = direction / d_norm
                lateral = body - body[0]
                lat_proj = lateral - np.outer(lateral @ d_hat, d_hat)
                out["lateral_deviation_max"] = float(np.max(np.linalg.norm(lat_proj, axis=1)))
            else:
                out["lateral_deviation_max"] = 0.0
        else:
            out["lateral_deviation_max"] = 0.0
        return out

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
        success_step_tolerance = float(obstacle_cfg.get("success_step_tolerance", 0.03))
        env_stone_margin = float(getattr(env, "stone_margin", 0.0))

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
        plat_xy = np.asarray(getattr(scene, "support_platforms", np.zeros((0, 4))), dtype=np.float32)
        river_x = np.asarray(scene.river_x, dtype=np.float32)
        has_river = bool(getattr(scene, "has_river", True))
        lmax = float(scene.l_max)
        target = np.asarray(scene.goal_mid, dtype=np.float32)
        body_step_limit = float(
            getattr(env, "body_step_norm_max", getattr(env, "body_shift_limit", lmax))
        )
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
            step_width=float(
                getattr(
                    env,
                    "step_width",
                    getattr(
                        env,
                        "stance_width",
                        getattr(
                            env,
                            "stance_width_nominal",
                            float(getattr(env, "y_L_nominal", 0.15)) - float(getattr(env, "y_R_nominal", -0.15)),
                        ),
                    ),
                )
            ),
            half_pair_length=float(
                getattr(
                    env,
                    "fore_hind_offset",
                    getattr(env, "fore_hind_nominal", abs(float(getattr(env, "x_f_nominal", 0.18)))),
                )
            ),
            centerline_y=float(getattr(env, "centerline_y", 0.0)),
            env=env,
        )
        goal_feet = env._nominal_feet(target, float(yaw[-1]))
        final_goal_error = float(np.linalg.norm(body[-1] - target))
        terminal_foot_error = np.asarray(
            [np.linalg.norm(feet[leg][-1] - goal_feet[leg]) for leg in ("FL", "FR", "RL", "RR")],
            dtype=np.float32,
        )

        foot_v_all = []
        for leg in ("FL", "FR", "RL", "RR"):
            foot_v_all.append(
                _support_violation(
                    feet[leg],
                    centers,
                    radii,
                    river_x,
                    has_river=has_river,
                    stone_margin=env_stone_margin,
                    platforms=plat_xy,
                )
            )
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
        if actions.ndim == 2 and actions.shape[1] == 12:
            body_a = np.linalg.norm(actions[:, :2], axis=1)
            alpha_ex = np.maximum(0.0, np.abs(actions[:, 2]) - float(getattr(env, "omega_max", 0.22)))
            doff_m = np.linalg.norm(actions[:, 3:11].reshape(-1, 4, 2), axis=-1).max(axis=1)
            dtau = actions[:, 11]
            phase_a = np.maximum(0.0, np.maximum(dtau - float(getattr(env, "phase_rate_limit", 1.0)), -dtau))
            body_lim = float(getattr(env, "body_step_norm_max", getattr(env, "body_shift_limit", body_step_limit)))
            doff_lim = float(
                getattr(env, "residual_rate_limit", getattr(env, "offset_rate_limit", 0.06))
            )
            action_step_v = np.maximum(
                0.0,
                np.maximum.reduce([body_a - body_lim, alpha_ex, doff_m - doff_lim, phase_a]),
            ).astype(np.float32)
        elif actions.ndim == 2 and actions.shape[1] >= 16:
            # 20D v2 env: [a_s,a_y, alpha, dd_residual(8), dx_f,dx_r, dy_L,dy_R, dtau]
            body_a = np.linalg.norm(actions[:, :2], axis=1)
            alpha_ex = np.maximum(0.0, np.abs(actions[:, 2]) - float(getattr(env, "omega_max", 0.22)))
            doff_m = np.linalg.norm(actions[:, 3:11].reshape(-1, 4, 2), axis=-1).max(axis=1)
            lx_t = float(getattr(env, "template_x_rate_limit", 0.08))
            ly_t = float(getattr(env, "template_y_rate_limit", 0.06))
            tmpl_v = np.maximum(0.0, np.abs(actions[:, 11]) - lx_t)
            tmpl_v = tmpl_v + np.maximum(0.0, np.abs(actions[:, 12]) - lx_t)
            tmpl_v = tmpl_v + np.maximum(0.0, np.abs(actions[:, 13]) - ly_t)
            tmpl_v = tmpl_v + np.maximum(0.0, np.abs(actions[:, 14]) - ly_t)
            dtau = actions[:, 15]
            phase_a = np.maximum(0.0, np.maximum(dtau - float(getattr(env, "phase_rate_limit", 1.0)), -dtau))
            body_lim = float(getattr(env, "body_step_norm_max", getattr(env, "body_shift_limit", body_step_limit)))
            doff_lim = float(
                getattr(env, "residual_rate_limit", getattr(env, "offset_rate_limit", 0.06))
            )
            action_step_v = np.maximum(
                0.0,
                np.maximum.reduce([body_a - body_lim, alpha_ex, doff_m - doff_lim, tmpl_v, phase_a]),
            ).astype(np.float32)
        elif actions.ndim == 2 and actions.shape[1] >= 8:
            # older 8D v2: [a_xy, alpha, dqL, dqR, dtau]
            body_a = np.linalg.norm(actions[:, :2], axis=1)
            fh_a = np.linalg.norm(actions[:, 3:7], axis=1)
            dtau = actions[:, 7]
            phase_a = np.maximum(
                0.0, np.maximum(dtau - float(getattr(env, "phase_rate_limit", 1.0)), -dtau)
            )
            fh_lim = float(getattr(env, "foothold_speed_limit", swing_step_limit))
            body_lim = float(getattr(env, "body_shift_limit", body_step_limit))
            action_step_v = np.maximum(
                0.0,
                np.maximum(np.maximum(body_a - body_lim, fh_a - fh_lim), phase_a),
            ).astype(np.float32)
        elif actions.ndim == 2 and actions.shape[1] >= 6:
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
            and step_v_max <= success_step_tolerance
            and terminal_foot_error_max <= float(getattr(env, "terminal_foot_margin", 0.20))
        )

        cand_costs = planning_result.get("candidate_costs", None)
        n_modes = int(len(cand_costs)) if cand_costs is not None else 1
        best_idx = int(planning_result.get("best_idx", 0))

        return {
            "success": success,
            "success_goal_margin": success_margin,
            "success_foothold_margin": foothold_margin,
            "success_step_tolerance": success_step_tolerance,
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
            # ── Follower-friendliness metrics ──
            **self._follower_metrics(body, yaw, feet, mode, lmax),
        }

