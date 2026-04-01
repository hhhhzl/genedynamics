from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except Exception:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]

from genedynamics.core.energy import EnergyTerm, LegacyEnergyFunctional
from genedynamics.envs.obstacles.stepping_stones import _platform_union_margin_np, foot_stepping_violation_np
from genedynamics.tasks.stepping_stones import SteppingStonesScene, sample_stepping_stones_scene

MODE_SUPPORT_FL_RR = 0
MODE_SUPPORT_FR_RL = 1
NUM_MODES = 2

LEG_ORDER = ("FL", "FR", "RL", "RR")
LEG_TO_IDX: Dict[str, int] = {n: i for i, n in enumerate(LEG_ORDER)}
LEFT_LEGS = ("FL", "RL")
RIGHT_LEGS = ("FR", "RR")

_SEP_PAIRS: Tuple[Tuple[int, int], ...] = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _to_np(x: Any, n: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size < n:
        arr = np.pad(arr, (0, n - arr.size), constant_values=0.0)
    return arr[:n]


def _wrap_np(theta: float) -> float:
    return float(np.arctan2(np.sin(theta), np.cos(theta)))


def _clip_norm_np(v: np.ndarray, limit: float) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n <= limit or n < 1e-8:
        return v
    return (float(limit) / n * v).astype(np.float32)


def _swing_support_leg_indices(mode: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return (swing_idx0, swing_idx1), (support_idx0, support_idx1) for LEG_ORDER indices."""
    if int(mode) == MODE_SUPPORT_FL_RR:
        return (1, 2), (0, 3)
    return (0, 3), (1, 2)


@dataclass
class QuadrupedSteppingStones2DEnv:
    """
    Stepping stones 2D: corridor body [s,ey] + fixed nominal template + per-foot residual (16D state).
    Template (x_f,x_r,y_L,y_R) is NOT in state — always nominal.

    State (16D):
      [ s, ey, epsi, vs, vy, omega,
        rFL_xy..rRR_xy (8),
        mode, tau ]

    Action (12D): [ body_xy_rate(2), yaw_rate, dres 8, dtau ]

    Residual rates are mode-conditioned: support legs use ``support_residual_rate_scale``;
    swing legs use full rate, scaled by phase (see ``phase_tau_split``).
    """

    dt: float = 1.0
    horizon: int = 12

    scene_seed: int = 0
    scene_level: int = 1
    scene: Optional[SteppingStonesScene] = None
    obstacles: Optional[Any] = None
    start_mid: Tuple[float, float] = (-1.25, 0.0)
    goal_mid: Tuple[float, float] = (1.25, 0.0)

    centerline_y: float = 0.0
    ey_max: float = 0.08
    epsi_max: float = 0.22
    vs_max: float = 0.28
    vy_max: float = 0.10
    omega_max: float = 0.22
    body_step_norm_max: float = 0.22
    allow_backward_slack: float = 0.005

    x_f_nominal: float = 0.18
    x_r_nominal: float = -0.18
    y_L_nominal: float = 0.15
    y_R_nominal: float = -0.15

    residual_x_limit: float = 0.05
    residual_y_limit: float = 0.05
    residual_rate_limit: float = 0.06
    support_residual_rate_scale: float = 0.06
    stone_margin: float = 0.032

    phase_tau_split: float = 0.7
    phase_swing_scale_late: float = 0.2
    rollover_swing_violation_max: float = 0.04
    tau_cap_when_blocked: float = 0.99

    min_foot_separation: float = 0.10
    support_dx_max: float = 0.14
    support_dy_max: float = 0.08

    phase_rate_limit: float = 0.45

    state_dim: int = 16
    act_dim: int = 12
    enable_cfs_safety_points_qp: bool = True
    # Opt-in hint for CFSQP: safety-point scenes should budget constraints per point, not per obstacle.
    cfs_qp_num_safety_points: int = 4
    # QP/CFS: use schedule margin only (no robot_radius) for foot-point constraints on thin stones.
    cfs_qp_safety_clearance_margin_only: bool = True
    # Plugin only: how ``jax_cfs_custom_safety_constraints`` picks a target foothold per foot.
    # ``legacy``: min distance among lane-valid (and forward for swing). Often picks far/negative-margin stones.
    # ``reachable``: prefer targets that already satisfy weighted clearance; else best margin; else legacy.
    cfs_custom_constraint_target_policy: str = "legacy"
    # Max forward x-gap (foot -> stone center) for swing-foot targets when policy is ``reachable``.
    # Initialized from scene ``l_max`` in ``__post_init__``; overridden if set > 0 before post_init ends.
    cfs_custom_swing_forward_cap: float = 0.0
    cfs_custom_swing_forward_slack: float = 0.08

    terminal_foot_penalty: float = 0.35
    terminal_foot_margin: float = 0.20
    terminal_reward_weight: float = 300.0

    task_terminal_foot_weight: float = 0.55

    def __post_init__(self) -> None:
        scene = self.scene
        if scene is None and self.obstacles is not None and hasattr(self.obstacles, "stepping_scene"):
            scene = getattr(self.obstacles, "stepping_scene")
        if scene is None:
            scene = sample_stepping_stones_scene(
                level=int(self.scene_level),
                seed=int(self.scene_seed),
                l_max=0.35,
                stance_width=float(abs(self.y_L_nominal - self.y_R_nominal)),
                start_mid=self.start_mid,
                goal_mid=self.goal_mid,
                fore_hind_offset=float(abs(self.x_f_nominal)),
            )
        self.scene = scene
        self.horizon = max(int(self.horizon), int(scene.k_horizon))
        self.target = np.asarray(scene.goal_mid, dtype=np.float32)
        self.start = np.asarray(scene.start_mid, dtype=np.float32)
        self.centerline_y = float(0.5 * (self.start[1] + self.target[1]))

        self._centers = np.asarray(self.scene.stones_centers, dtype=np.float32)
        self._radii = np.asarray(self.scene.stones_radii, dtype=np.float32)
        self._support_platforms = np.asarray(
            getattr(self.scene, "support_platforms", np.zeros((0, 4), dtype=np.float32)),
            dtype=np.float32,
        )
        self._n_sp = int(self._support_platforms.shape[0])
        self._support_platforms_pad = np.zeros((4, 4), dtype=np.float32)
        if self._n_sp > 0:
            self._support_platforms_pad[: self._n_sp] = self._support_platforms[:4]
        self._n_stones = int(self._centers.shape[0])
        if self._n_stones == 0 and self._n_sp == 0:
            raise ValueError("Scene must contain at least one stone or support platform.")

        y0, y1 = float(self.scene.map_y[0]), float(self.scene.map_y[1])
        rx0, rx1 = float(self.scene.river_x[0]), float(self.scene.river_x[1])
        self._river_center_np = np.asarray([0.5 * (rx0 + rx1), 0.5 * (y0 + y1)], dtype=np.float32)
        self._river_half_np = np.asarray([0.5 * abs(rx1 - rx0), 0.5 * abs(y1 - y0)], dtype=np.float32)
        self._has_river_scene = bool(getattr(self.scene, "has_river", False))
        # Foothold candidates for stepping-specific CFS constraints:
        # stones + support platforms, each with center/radius/lane tag.
        cand_centers = []
        cand_radii = []
        cand_lane = []  # +1 left, -1 right, 0 neutral
        for c, r in zip(self._centers, self._radii):
            c2 = np.asarray(c, dtype=np.float32).reshape(2)
            cand_centers.append(c2)
            cand_radii.append(float(r))
            dy = float(c2[1] - self.centerline_y)
            cand_lane.append(1.0 if dy > 1e-6 else (-1.0 if dy < -1e-6 else 0.0))
        for rect in self._support_platforms:
            x0r, x1r, y0r, y1r = map(float, np.asarray(rect, dtype=np.float32).reshape(4))
            c2 = np.asarray([0.5 * (x0r + x1r), 0.5 * (y0r + y1r)], dtype=np.float32)
            rr = max(1e-4, 0.5 * min(abs(x1r - x0r), abs(y1r - y0r)))
            cand_centers.append(c2)
            cand_radii.append(float(rr))
            cand_lane.append(0.0)
        self._foothold_centers_np = np.asarray(cand_centers, dtype=np.float32).reshape(-1, 2)
        self._foothold_radii_np = np.asarray(cand_radii, dtype=np.float32).reshape(-1)
        self._foothold_lane_np = np.asarray(cand_lane, dtype=np.float32).reshape(-1)
        if self._foothold_centers_np.shape[0] == 0:
            self._foothold_centers_np = np.zeros((1, 2), dtype=np.float32)
            self._foothold_radii_np = np.ones((1,), dtype=np.float32) * 1e-3
            self._foothold_lane_np = np.zeros((1,), dtype=np.float32)

        lm = float(getattr(self.scene, "l_max", 0.35))
        if float(self.cfs_custom_swing_forward_cap) <= 0.0:
            self.cfs_custom_swing_forward_cap = float(lm + float(self.cfs_custom_swing_forward_slack))

        s0 = float(self.start[0])
        ey0 = float(self.start[1] - self.centerline_y)
        epsi0 = float(np.arctan2(self.target[1] - self.start[1], self.target[0] - self.start[0]))

        self._default_state = self._pack_state(
            s=s0,
            ey=ey0,
            epsi=epsi0,
            vel=np.zeros((3,), dtype=np.float32),
            residuals=np.zeros((4, 2), dtype=np.float32),
            mode=MODE_SUPPORT_FL_RR,
            tau=0.0,
        )

    def _body_world_np(self, s: float, ey: float) -> np.ndarray:
        return np.asarray([s, self.centerline_y + ey], dtype=np.float32)

    def _template_local_np(self) -> Dict[str, np.ndarray]:
        return {
            "FL": np.asarray([self.x_f_nominal, self.y_L_nominal], dtype=np.float32),
            "FR": np.asarray([self.x_f_nominal, self.y_R_nominal], dtype=np.float32),
            "RL": np.asarray([self.x_r_nominal, self.y_L_nominal], dtype=np.float32),
            "RR": np.asarray([self.x_r_nominal, self.y_R_nominal], dtype=np.float32),
        }

    def _pack_state(
        self,
        *,
        s: float,
        ey: float,
        epsi: float,
        vel: np.ndarray,
        residuals: np.ndarray,
        mode: int,
        tau: float,
    ) -> np.ndarray:
        x = np.zeros((self.state_dim,), dtype=np.float32)
        x[0] = float(s)
        x[1] = float(ey)
        x[2] = float(epsi)
        x[3:6] = np.asarray(vel, dtype=np.float32)
        x[6:14] = np.asarray(residuals, dtype=np.float32).reshape(8)
        x[14] = float(int(mode) % NUM_MODES)
        x[15] = float(np.clip(tau, 0.0, 1.0))
        return x

    def _decode_state(self, state: np.ndarray):
        x = _to_np(state, self.state_dim)
        s = float(x[0])
        ey = float(x[1])
        epsi = float(x[2])
        vel = x[3:6].astype(np.float32)
        residuals = x[6:14].reshape(4, 2).astype(np.float32)
        mode = int(np.rint(float(x[14]))) % NUM_MODES
        tau = float(np.clip(x[15], 0.0, 1.0))
        return s, ey, epsi, vel, residuals, mode, tau

    def _decode_feet_np(self, s: float, ey: float, epsi: float, residuals: np.ndarray) -> Dict[str, np.ndarray]:
        body = self._body_world_np(s, ey)
        tpl = self._template_local_np()
        R = np.asarray([[np.cos(epsi), -np.sin(epsi)], [np.sin(epsi), np.cos(epsi)]], dtype=np.float32)
        feet: Dict[str, np.ndarray] = {}
        for i, leg in enumerate(LEG_ORDER):
            local = tpl[leg] + residuals[i]
            feet[leg] = (body + R @ local).astype(np.float32)
        return feet

    def _feet_from_state_np(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        s, ey, epsi, _vel, res, _mode, _tau = self._decode_state(state)
        return self._decode_feet_np(s, ey, epsi, res)

    def _nominal_feet(self, body_world: np.ndarray, psi: float) -> Dict[str, np.ndarray]:
        body_world = np.asarray(body_world, dtype=np.float32).reshape(2)
        psi = float(psi)
        tpl = self._template_local_np()
        c, s = float(np.cos(psi)), float(np.sin(psi))
        R = np.asarray([[c, -s], [s, c]], dtype=np.float32)
        return {leg: (body_world + R @ tpl[leg]).astype(np.float32) for leg in LEG_ORDER}

    def _support_pair_legs(self, mode: int) -> Tuple[str, str]:
        return ("FL", "RR") if int(mode) == MODE_SUPPORT_FL_RR else ("FR", "RL")

    def _support_center_np(self, feet: Dict[str, np.ndarray], mode: int) -> np.ndarray:
        a, b = self._support_pair_legs(mode)
        return (0.5 * (feet[a] + feet[b])).astype(np.float32)

    def _support_violation_np(self, p: np.ndarray) -> float:
        return foot_stepping_violation_np(
            p,
            self._centers,
            self._radii,
            self._support_platforms,
            float(self.stone_margin),
        )

    def _swing_foot_violation_max_np(self, feet: Dict[str, np.ndarray], mode: int) -> float:
        swing_legs = _swing_support_leg_indices(mode)[0]
        a, b = LEG_ORDER[swing_legs[0]], LEG_ORDER[swing_legs[1]]
        return max(self._support_violation_np(feet[a]), self._support_violation_np(feet[b]))

    def _phase_swing_scale_np(self, tau: float) -> float:
        if tau < float(self.phase_tau_split):
            return 1.0
        denom = max(1e-6, 1.0 - float(self.phase_tau_split))
        return float(self.phase_swing_scale_late + (1.0 - self.phase_swing_scale_late) * (1.0 - tau) / denom)

    def _mask_residual_rates_np(self, dres: np.ndarray, mode: int, tau: float) -> np.ndarray:
        (is0, is1), (ip0, ip1) = _swing_support_leg_indices(mode)
        out = dres.astype(np.float32).copy()
        sw_scale = self._phase_swing_scale_np(tau)
        sup_scale = float(self.support_residual_rate_scale)
        for i in range(4):
            if i in (is0, is1):
                out[i] *= sw_scale
        else:
                out[i] *= sup_scale
        return out

    def _commit_residuals_on_rollover_np(self, res: np.ndarray, mode: int) -> np.ndarray:
        _, (ip0, ip1) = _swing_support_leg_indices(mode)
        out = res.astype(np.float32).copy()
        out[ip0, :] = 0.0
        out[ip1, :] = 0.0
        return out

    def reset(self, rng: Optional[Any] = None, seed: Optional[int] = None, **kwargs: Any):
        _ = (rng, seed, kwargs)
        return self._default_state.copy(), {}

    def _apply_min_foot_separation_np(self, residual: np.ndarray, feet: Dict[str, np.ndarray]) -> np.ndarray:
        next_r = residual.astype(np.float32).copy()
        lx, ly = float(self.residual_x_limit), float(self.residual_y_limit)
        for i, j in _SEP_PAIRS:
            a, b = LEG_ORDER[i], LEG_ORDER[j]
            d = float(np.linalg.norm(feet[a] - feet[b]))
            if d >= self.min_foot_separation:
                continue
            if (a in LEFT_LEGS and b in RIGHT_LEGS) or (a in RIGHT_LEGS and b in LEFT_LEGS):
                next_r[i, 1] = float(np.clip(next_r[i, 1] + 0.5 * ly, -ly, ly))
                next_r[j, 1] = float(np.clip(next_r[j, 1] - 0.5 * ly, -ly, ly))
            else:
                next_r[i, 0] = float(np.clip(next_r[i, 0] - 0.5 * lx, -lx, lx))
                next_r[j, 0] = float(np.clip(next_r[j, 0] + 0.5 * lx, -lx, lx))
        return next_r

    def transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        s, ey, epsi, _vel, residual, mode, tau = self._decode_state(state)
        u = _to_np(action, self.act_dim)

        body_step = _clip_norm_np(u[:2], self.body_step_norm_max)
        a_s = float(np.clip(body_step[0], -self.allow_backward_slack, self.vs_max))
        a_y = float(np.clip(body_step[1], -self.vy_max, self.vy_max))
        alpha = float(np.clip(u[2], -self.omega_max, self.omega_max))

        dres = np.asarray(u[3:11], dtype=np.float32).reshape(4, 2)
        dres = np.stack([_clip_norm_np(dres[i], self.residual_rate_limit) for i in range(4)], axis=0)
        dres = self._mask_residual_rates_np(dres, mode, tau)
        dtau = float(np.clip(u[11], 0.0, self.phase_rate_limit))

        next_s = max(s - self.allow_backward_slack, s + self.dt * a_s)
        next_ey = float(np.clip(ey + self.dt * a_y, -self.ey_max, self.ey_max))
        next_epsi = float(np.clip(_wrap_np(epsi + self.dt * alpha), -self.epsi_max, self.epsi_max))
        next_vs = float(
            np.clip((next_s - s) / max(self.dt, 1e-6), -self.allow_backward_slack / max(self.dt, 1e-6), self.vs_max)
        )
        next_vy = float(np.clip((next_ey - ey) / max(self.dt, 1e-6), -self.vy_max, self.vy_max))
        next_omega = float(np.clip((next_epsi - epsi) / max(self.dt, 1e-6), -self.omega_max, self.omega_max))

        next_res = residual + self.dt * dres
        next_res[:, 0] = np.clip(next_res[:, 0], -self.residual_x_limit, self.residual_x_limit)
        next_res[:, 1] = np.clip(next_res[:, 1], -self.residual_y_limit, self.residual_y_limit)

        feet = self._decode_feet_np(next_s, next_ey, next_epsi, next_res)
        support_center = self._support_center_np(feet, mode)
        body_world = self._body_world_np(next_s, next_ey)
        db = body_world - support_center
        db[0] = float(np.clip(db[0], -self.support_dx_max, self.support_dx_max))
        db[1] = float(np.clip(db[1], -self.support_dy_max, self.support_dy_max))
        body_world = (support_center + db).astype(np.float32)
        next_s = max(s - self.allow_backward_slack, float(body_world[0]))
        next_ey = float(np.clip(body_world[1] - self.centerline_y, -self.ey_max, self.ey_max))

        feet = self._decode_feet_np(next_s, next_ey, next_epsi, next_res)
        next_res = self._apply_min_foot_separation_np(next_res, feet)

        proposed_tau = float(np.clip(tau + self.dt * dtau, 0.0, 1.0))
        next_mode = int(mode)
        next_tau = proposed_tau
        sw_v = self._swing_foot_violation_max_np(feet, mode)
        can_roll = sw_v <= float(self.rollover_swing_violation_max) + 1e-6

        if proposed_tau >= 1.0 - 1e-7:
            if can_roll:
                next_tau = 0.0
                next_mode = (int(mode) + 1) % NUM_MODES
                next_res = self._commit_residuals_on_rollover_np(next_res, mode)
            else:
                next_tau = min(proposed_tau, float(self.tau_cap_when_blocked))

        return self._pack_state(
            s=next_s,
            ey=next_ey,
            epsi=next_epsi,
            vel=np.asarray([next_vs, next_vy, next_omega], dtype=np.float32),
            residuals=next_res,
            mode=next_mode,
            tau=next_tau,
        )

    def rollout_actions(self, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        x = _to_np(state, self.state_dim)
        traj = [x.copy()]
        for a in np.asarray(actions, dtype=np.float32):
            x = self.transition(x, a)
            traj.append(x.copy())
        return np.asarray(traj, dtype=np.float32)

    def model_transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self.transition(state, action)

    def get_safety_points(self, state: np.ndarray) -> np.ndarray:
        feet = self._feet_from_state_np(state)
        return np.stack([feet[leg] for leg in LEG_ORDER], axis=0).astype(np.float32)

    def get_jacobian_safety_points_action_local(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x = _to_np(state, self.state_dim)
        u = _to_np(action, self.act_dim)
        eps = 1e-3
        J = np.zeros((4, 2, self.act_dim), dtype=np.float32)
        for i in range(self.act_dim):
            du = np.zeros((self.act_dim,), dtype=np.float32)
            du[i] = eps
            pp = self.get_safety_points(self.transition(x, u + du))
            pm = self.get_safety_points(self.transition(x, u - du))
            J[:, :, i] = ((pp - pm) / (2.0 * eps)).astype(np.float32)
        return J

    def get_jacobian_safety_points_action(self, state: np.ndarray) -> np.ndarray:
        zero_u = np.zeros((self.act_dim,), dtype=np.float32)
        return self.get_jacobian_safety_points_action_local(state, zero_u)

    def terminal_distance(self, state: np.ndarray) -> float:
        s, ey, epsi, _vel, res, mode, _tau = self._decode_state(state)
        body = self._body_world_np(s, ey)
        feet = self._decode_feet_np(s, ey, epsi, res)
        body_err = float(np.linalg.norm(body - self.target))
        support_center = self._support_center_np(feet, mode)
        sync_err = float(np.linalg.norm(body - support_center))
        foot_err = float(np.mean([self._support_violation_np(feet[leg]) for leg in LEG_ORDER]))
        return body_err + 0.5 * sync_err + float(self.terminal_foot_penalty) * foot_err

    def terminal_distance_jax(self, state: Any) -> Any:
        if jnp is None:
            raise RuntimeError("terminal_distance_jax requires JAX.")
        x = jnp.asarray(state, dtype=jnp.float32).reshape(-1)[: self.state_dim]
        s, ey, epsi = x[0], x[1], x[2]
        res = x[6:14].reshape(4, 2)
        mode = jnp.mod(jnp.rint(x[14]).astype(jnp.int32), NUM_MODES)
        body = jnp.stack([s, self.centerline_y + ey], dtype=jnp.float32)
        c = jnp.cos(epsi)
        si = jnp.sin(epsi)
        R = jnp.asarray([[c, -si], [si, c]], dtype=jnp.float32)
        tpl = jnp.stack(
            [
                jnp.asarray([self.x_f_nominal, self.y_L_nominal], dtype=jnp.float32),
                jnp.asarray([self.x_f_nominal, self.y_R_nominal], dtype=jnp.float32),
                jnp.asarray([self.x_r_nominal, self.y_L_nominal], dtype=jnp.float32),
                jnp.asarray([self.x_r_nominal, self.y_R_nominal], dtype=jnp.float32),
            ],
            axis=0,
        )
        local = tpl + res
        feet4 = body[None, :] + local @ R.T
        goal_mid_j = jnp.asarray(self.target, dtype=jnp.float32)
        body_err = jnp.linalg.norm(body - goal_mid_j)
        sc = jnp.where(mode == MODE_SUPPORT_FL_RR, 0.5 * (feet4[0] + feet4[3]), 0.5 * (feet4[1] + feet4[2]))
        sync_err = jnp.linalg.norm(body - sc)
        centers_j = jnp.asarray(self._centers, dtype=jnp.float32)
        radii_j = jnp.asarray(self._radii, dtype=jnp.float32)
        margin_j = jnp.asarray(self.stone_margin, dtype=jnp.float32)
        plat_pad = jnp.asarray(self._support_platforms_pad, dtype=jnp.float32)
        nsp = int(self._n_sp)

        def one_v(ft):
            if centers_j.shape[0] > 0:
                d = jnp.linalg.norm(centers_j - ft[None, :], axis=-1)
                vd = jnp.min(jnp.maximum(d - (radii_j - margin_j), 0.0))
            else:
                vd = jnp.asarray(1e9, dtype=jnp.float32)

            def pb(bounds):
                xm, xM, ym, yM = bounds[0], bounds[1], bounds[2], bounds[3]
                xl = xm + margin_j
                xR = xM - margin_j
                yb = ym + margin_j
                yT = yM - margin_j
                ins = (ft[0] >= xl) & (ft[0] <= xR) & (ft[1] >= yb) & (ft[1] <= yT)
                dx = jnp.maximum(0.0, xl - ft[0]) + jnp.maximum(0.0, ft[0] - xR)
                dy = jnp.maximum(0.0, yb - ft[1]) + jnp.maximum(0.0, ft[1] - yT)
                dist = jnp.sqrt(dx * dx + dy * dy)
                return jnp.where(ins, 0.0, dist)

            p0, p1, p2, p3 = pb(plat_pad[0]), pb(plat_pad[1]), pb(plat_pad[2]), pb(plat_pad[3])
            stacked = jnp.stack([p0, p1, p2, p3])
            msk = jnp.array([1.0 if k < nsp else 0.0 for k in range(4)], dtype=jnp.float32)
            vp = jnp.min(jnp.where(msk > 0.5, stacked, jnp.asarray(1e9, dtype=jnp.float32)))
            return jnp.minimum(vd, vp)

        foot_err = 0.25 * (one_v(feet4[0]) + one_v(feet4[1]) + one_v(feet4[2]) + one_v(feet4[3]))
        psi_ref = jnp.arctan2(goal_mid_j[1] - body[1], goal_mid_j[0] - body[0])
        c_g = jnp.cos(psi_ref)
        s_g = jnp.sin(psi_ref)
        Rg = jnp.asarray([[c_g, -s_g], [s_g, c_g]], dtype=jnp.float32)
        tpl_goal = jnp.stack(
            [
                jnp.asarray([self.x_f_nominal, self.y_L_nominal], dtype=jnp.float32),
                jnp.asarray([self.x_f_nominal, self.y_R_nominal], dtype=jnp.float32),
                jnp.asarray([self.x_r_nominal, self.y_L_nominal], dtype=jnp.float32),
                jnp.asarray([self.x_r_nominal, self.y_R_nominal], dtype=jnp.float32),
            ],
            axis=0,
        )
        goal_feet = goal_mid_j[None, :] + tpl_goal @ Rg.T
        arrange_err = jnp.mean(
            jnp.stack(
                [
                    jnp.sum((feet4[i] - goal_feet[i]) ** 2)
                    for i in range(4)
                ]
            )
        )
        return (
            body_err
            + 0.5 * sync_err
            + float(self.terminal_foot_penalty) * foot_err
            + 0.15 * jnp.sqrt(arrange_err)
        )

    def jax_safety_points(self, state: Any) -> Any:
        if jnp is None:
            raise RuntimeError("jax_safety_points requires JAX.")
        x = jnp.asarray(state, dtype=jnp.float32).reshape(-1)[: self.state_dim]
        s, ey, epsi = x[0], x[1], x[2]
        res = x[6:14].reshape(4, 2)
        body = jnp.asarray([s, self.centerline_y + ey], dtype=jnp.float32)
        c = jnp.cos(epsi)
        s_ = jnp.sin(epsi)
        R = jnp.asarray([[c, -s_], [s_, c]], dtype=jnp.float32)
        tpl = jnp.stack(
            [
                jnp.asarray([self.x_f_nominal, self.y_L_nominal], dtype=jnp.float32),
                jnp.asarray([self.x_f_nominal, self.y_R_nominal], dtype=jnp.float32),
                jnp.asarray([self.x_r_nominal, self.y_L_nominal], dtype=jnp.float32),
                jnp.asarray([self.x_r_nominal, self.y_R_nominal], dtype=jnp.float32),
            ],
            axis=0,
        )
        local = tpl + res
        return body[None, :] + local @ R.T

    def jax_model_transition(self, state: Any, action: Any) -> Any:
        if jax is None or jnp is None:
            raise RuntimeError("jax_model_transition requires JAX.")

        def clip_norm(v, limit):
            n = jnp.linalg.norm(v)
            scale = jnp.where(n > limit, limit / jnp.maximum(n, 1e-8), 1.0)
            return v * scale

        pt_split = float(self.phase_tau_split)
        swing_late = float(self.phase_swing_scale_late)
        sup_scale = float(self.support_residual_rate_scale)
        roll_vmax = float(self.rollover_swing_violation_max)
        tau_cap = float(self.tau_cap_when_blocked)
        xfn, xrn = float(self.x_f_nominal), float(self.x_r_nominal)
        yLn, yRn = float(self.y_L_nominal), float(self.y_R_nominal)

        x = jnp.asarray(state, dtype=jnp.float32).reshape(-1)[: self.state_dim]
        u = jnp.asarray(action, dtype=jnp.float32).reshape(-1)[: self.act_dim]

        s, ey, epsi = x[0], x[1], x[2]
        res = x[6:14].reshape(4, 2)
        mode = jnp.mod(jnp.rint(x[14]).astype(jnp.int32), NUM_MODES)
        tau0 = jnp.clip(x[15], 0.0, 1.0)

        step = clip_norm(u[:2], self.body_step_norm_max)
        a_s = jnp.clip(step[0], -self.allow_backward_slack, self.vs_max)
        a_y = jnp.clip(step[1], -self.vy_max, self.vy_max)
        alpha = jnp.clip(u[2], -self.omega_max, self.omega_max)
        dres_raw = u[3:11].reshape(4, 2)
        dres = jnp.stack([clip_norm(dres_raw[k], self.residual_rate_limit) for k in range(4)], axis=0)

        is_m0 = mode == MODE_SUPPORT_FL_RR
        swing_m = jnp.where(is_m0, jnp.array([0.0, 1.0, 1.0, 0.0], dtype=jnp.float32), jnp.array([1.0, 0.0, 0.0, 1.0], dtype=jnp.float32))
        sw_scale = jnp.where(tau0 < pt_split, 1.0, swing_late + (1.0 - swing_late) * (1.0 - tau0) / jnp.maximum(1e-6, 1.0 - pt_split))
        row_scale = swing_m * sw_scale + (1.0 - swing_m) * sup_scale
        dres = dres * row_scale[:, None]

        dtau = jnp.clip(u[11], 0.0, self.phase_rate_limit)

        next_s = jnp.maximum(s - self.allow_backward_slack, s + self.dt * a_s)
        next_ey = jnp.clip(ey + self.dt * a_y, -self.ey_max, self.ey_max)
        next_epsi = jnp.clip(
            jnp.arctan2(jnp.sin(epsi + self.dt * alpha), jnp.cos(epsi + self.dt * alpha)),
            -self.epsi_max,
            self.epsi_max,
        )
        next_vs = jnp.clip(
            (next_s - s) / jnp.maximum(self.dt, 1e-6),
            -self.allow_backward_slack / jnp.maximum(self.dt, 1e-6),
            self.vs_max,
        )
        next_vy = jnp.clip((next_ey - ey) / jnp.maximum(self.dt, 1e-6), -self.vy_max, self.vy_max)
        next_omega = jnp.clip((next_epsi - epsi) / jnp.maximum(self.dt, 1e-6), -self.omega_max, self.omega_max)

        next_res = res + self.dt * dres
        next_res = next_res.at[:, 0].set(jnp.clip(next_res[:, 0], -self.residual_x_limit, self.residual_x_limit))
        next_res = next_res.at[:, 1].set(jnp.clip(next_res[:, 1], -self.residual_y_limit, self.residual_y_limit))

        body_xy = jnp.stack([next_s, self.centerline_y + next_ey], dtype=jnp.float32)
        c = jnp.cos(next_epsi)
        s_ = jnp.sin(next_epsi)
        R = jnp.asarray([[c, -s_], [s_, c]], dtype=jnp.float32)
        tpl = jnp.stack(
            [
                jnp.stack([jnp.asarray(xfn), jnp.asarray(yLn)]),
                jnp.stack([jnp.asarray(xfn), jnp.asarray(yRn)]),
                jnp.stack([jnp.asarray(xrn), jnp.asarray(yLn)]),
                jnp.stack([jnp.asarray(xrn), jnp.asarray(yRn)]),
            ],
            axis=0,
        )
        local = tpl + next_res
        feet4 = body_xy[None, :] + local @ R.T

        sc = jnp.where(mode == MODE_SUPPORT_FL_RR, 0.5 * (feet4[0] + feet4[3]), 0.5 * (feet4[1] + feet4[2]))
        db = body_xy - sc
        db0 = jnp.clip(db[0], -self.support_dx_max, self.support_dx_max)
        db1 = jnp.clip(db[1], -self.support_dy_max, self.support_dy_max)
        body2 = sc + jnp.stack([db0, db1])
        next_es = jnp.maximum(s - self.allow_backward_slack, body2[0])
        next_eey = jnp.clip(body2[1] - self.centerline_y, -self.ey_max, self.ey_max)
        body_xy2 = jnp.stack([next_es, self.centerline_y + next_eey], dtype=jnp.float32)
        feet4 = body_xy2[None, :] + local @ R.T

        lx_f = float(self.residual_x_limit)
        ly_f = float(self.residual_y_limit)

        def sep_pair(nr: Any, ii: int, jj: int) -> Any:
            d = jnp.linalg.norm(feet4[ii] - feet4[jj])
            too = d < self.min_foot_separation
            lr = (ii in (0, 2) and jj in (1, 3)) or (ii in (1, 3) and jj in (0, 2))
            if lr:
                ni = jnp.clip(nr[ii, 1] + 0.5 * ly_f, -ly_f, ly_f)
                nj = jnp.clip(nr[jj, 1] - 0.5 * ly_f, -ly_f, ly_f)
                nr2 = nr.at[ii, 1].set(ni).at[jj, 1].set(nj)
            else:
                ni = jnp.clip(nr[ii, 0] - 0.5 * lx_f, -lx_f, lx_f)
                nj = jnp.clip(nr[jj, 0] + 0.5 * lx_f, -lx_f, lx_f)
                nr2 = nr.at[ii, 0].set(ni).at[jj, 0].set(nj)
            return jnp.where(too, nr2, nr)

        nr = next_res
        for ii, jj in _SEP_PAIRS:
            nr = sep_pair(nr, ii, jj)
        next_res = nr

        centers_j = jnp.asarray(self._centers, dtype=jnp.float32)
        radii_j = jnp.asarray(self._radii, dtype=jnp.float32)
        margin_j = jnp.asarray(self.stone_margin, dtype=jnp.float32)
        plat_pad = jnp.asarray(self._support_platforms_pad, dtype=jnp.float32)
        nsp = int(self._n_sp)

        def stone_v_of(pt):
            if centers_j.shape[0] > 0:
                d = jnp.linalg.norm(centers_j - pt[None, :], axis=-1)
                vd = jnp.min(jnp.maximum(d - (radii_j - margin_j), 0.0))
            else:
                vd = jnp.asarray(1e9, dtype=jnp.float32)

            def pb(bounds):
                xm, xM, ym, yM = bounds[0], bounds[1], bounds[2], bounds[3]
                xl = xm + margin_j
                xR = xM - margin_j
                yb = ym + margin_j
                yT = yM - margin_j
                ins = (pt[0] >= xl) & (pt[0] <= xR) & (pt[1] >= yb) & (pt[1] <= yT)
                dx = jnp.maximum(0.0, xl - pt[0]) + jnp.maximum(0.0, pt[0] - xR)
                dy = jnp.maximum(0.0, yb - pt[1]) + jnp.maximum(0.0, pt[1] - yT)
                dist = jnp.sqrt(dx * dx + dy * dy)
                return jnp.where(ins, 0.0, dist)

            p0, p1, p2, p3 = pb(plat_pad[0]), pb(plat_pad[1]), pb(plat_pad[2]), pb(plat_pad[3])
            stacked = jnp.stack([p0, p1, p2, p3])
            msk = jnp.array([1.0 if k < nsp else 0.0 for k in range(4)], dtype=jnp.float32)
            vp = jnp.min(jnp.where(msk > 0.5, stacked, jnp.asarray(1e9, dtype=jnp.float32)))
            return jnp.minimum(vd, vp)

        sw0 = jnp.where(is_m0, 1, 0)
        sw1 = jnp.where(is_m0, 2, 3)
        vsw = jnp.maximum(stone_v_of(feet4[sw0]), stone_v_of(feet4[sw1]))
        can_roll = vsw <= roll_vmax

        proposed_tau = jnp.clip(tau0 + self.dt * dtau, 0.0, 1.0)
        wants_roll = proposed_tau >= 1.0 - 1e-6
        roll_ok = jnp.logical_and(wants_roll, can_roll)
        blocked = jnp.logical_and(wants_roll, jnp.logical_not(can_roll))
        next_tau_f = jnp.where(roll_ok, 0.0, jnp.where(blocked, jnp.minimum(proposed_tau, tau_cap), proposed_tau))
        next_mode_f = jnp.where(roll_ok, jnp.mod(mode + 1, NUM_MODES), mode)

        sup0 = jnp.where(is_m0, 0, 1)
        sup1 = jnp.where(is_m0, 3, 2)
        committed = next_res
        committed = jnp.where(roll_ok, committed.at[sup0].set(0.0).at[sup1].set(0.0), committed)
        next_res_f = committed

        out = jnp.zeros((self.state_dim,), dtype=jnp.float32)
        out = out.at[0].set(next_es)
        out = out.at[1].set(next_eey)
        out = out.at[2].set(next_epsi)
        out = out.at[3:6].set(jnp.stack([next_vs, next_vy, next_omega]))
        out = out.at[6:14].set(next_res_f.reshape(8))
        out = out.at[14].set(next_mode_f.astype(jnp.float32))
        out = out.at[15].set(next_tau_f)
        return out

    def jax_transition(self, state: Any, action: Any) -> Any:
        """Compatibility alias used by CFSQP JAX rollout."""
        return self.jax_model_transition(state, action)

    def jax_jacobian_safety_points_action(self, state: Any) -> Any:
        if jax is None or jnp is None:
            raise RuntimeError("jax_jacobian_safety_points_action requires JAX.")
        x = jnp.asarray(state, dtype=jnp.float32).reshape(-1)[: self.state_dim]
        u0 = jnp.zeros((self.act_dim,), dtype=jnp.float32)

        def feet_next(u):
            return self.jax_safety_points(self.jax_model_transition(x, u))

        return jax.jacfwd(feet_next)(u0).astype(jnp.float32)

    def jax_jacobian_safety_points_action_local(self, state: Any, action: Any) -> Any:
        if jax is None or jnp is None:
            raise RuntimeError("jax_jacobian_safety_points_action_local requires JAX.")
        x = jnp.asarray(state, dtype=jnp.float32).reshape(-1)[: self.state_dim]
        u0 = jnp.asarray(action, dtype=jnp.float32).reshape(-1)[: self.act_dim]

        def feet_next(u):
            return self.jax_safety_points(self.jax_model_transition(x, u))

        return jax.jacfwd(feet_next)(u0).astype(jnp.float32)

    def _numpy_scene_qp_sdf(self, pt: np.ndarray) -> float:
        """Same convention as ``SteppingStonesForbiddenRegionObstacle.sdf``: larger = safer for CFS/QP."""
        p = np.asarray(pt, dtype=np.float32).reshape(2)
        if self._centers.shape[0] > 0:
            d = np.linalg.norm(self._centers - p[None, :], axis=-1)
            disk_m = float(np.max(self._radii - d))
        else:
            disk_m = float("-inf")
        plat_m = float(_platform_union_margin_np(p[None, :], self._support_platforms)[0])
        safe_margin = float(max(disk_m, plat_m))
        if self._has_river_scene:
            c = self._river_center_np
            h = self._river_half_np
            q = np.abs(p - c) - h
            outside = float(np.linalg.norm(np.maximum(q, 0.0)))
            inside = float(np.minimum(np.max(q), 0.0))
            sdf_river = outside + inside
            return float(min(safe_margin, sdf_river))
        return safe_margin

    def _jax_scene_qp_sdf(self, pt: Any) -> Any:
        if jnp is None:
            raise RuntimeError("JAX required.")
        p2 = jnp.asarray(pt, dtype=jnp.float32).reshape(2)[:2]
        centers_j = jnp.asarray(self._centers, dtype=jnp.float32)
        radii_j = jnp.asarray(self._radii, dtype=jnp.float32)
        plat_pad = jnp.asarray(self._support_platforms_pad, dtype=jnp.float32)
        nsp = int(self._n_sp)

        def _plat_margin_one(p):
            def one_bounds(bounds):
                xm, xM, ym, yM = bounds[0], bounds[1], bounds[2], bounds[3]
                return jnp.minimum(
                    jnp.minimum(p[0] - xm, xM - p[0]),
                    jnp.minimum(p[1] - ym, yM - p[1]),
                )

            m0 = one_bounds(plat_pad[0])
            m1 = one_bounds(plat_pad[1])
            m2 = one_bounds(plat_pad[2])
            m3 = one_bounds(plat_pad[3])
            stacked = jnp.stack([m0, m1, m2, m3])
            mask = jnp.array([1.0 if k < nsp else 0.0 for k in range(4)], dtype=jnp.float32)
            masked = jnp.where(mask > 0.5, stacked, -1e9)
            return jnp.max(masked)

        if centers_j.shape[0] > 0:
            dists = jnp.linalg.norm(centers_j - p2[None, :], axis=-1)
            disk_m = jnp.max(radii_j - dists)
        else:
            disk_m = jnp.asarray(-jnp.inf, dtype=jnp.float32)
        plat_vals = _plat_margin_one(p2)
        safe_margin = jnp.maximum(disk_m, plat_vals)

        has_r = jnp.asarray(self._has_river_scene, dtype=jnp.bool_)
        center = jnp.asarray(self._river_center_np, dtype=jnp.float32)
        half = jnp.asarray(self._river_half_np, dtype=jnp.float32)
        q = jnp.abs(p2 - center) - half
        outside = jnp.linalg.norm(jnp.maximum(q, 0.0))
        inside = jnp.minimum(jnp.max(q), 0.0)
        sdf_river = outside + inside
        return jnp.where(has_r, jnp.minimum(safe_margin, sdf_river), safe_margin)

    def numpy_cfs_safety_point_weights(self, state: np.ndarray) -> np.ndarray:
        x = _to_np(state, self.state_dim)
        mode = int(np.rint(float(x[14]))) % NUM_MODES
        tau = float(np.clip(x[15], 0.0, 1.0))
        swing_m = np.array(
            [0.0, 1.0, 1.0, 0.0] if mode == MODE_SUPPORT_FL_RR else [1.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
        pt_split = float(self.phase_tau_split)
        swing_late = float(self.phase_swing_scale_late)
        denom = max(1e-6, 1.0 - pt_split)
        if tau < pt_split:
            swing_clearance_scale = swing_late
        else:
            swing_clearance_scale = swing_late + (1.0 - swing_late) * (tau - pt_split) / denom
        return (1.0 - swing_m) + swing_m * np.float32(swing_clearance_scale)

    def numpy_cfs_alm_g_plus_from_state(self, state: np.ndarray, clearance: Any) -> float:
        feet = self._feet_from_state_np(state)
        w = self.numpy_cfs_safety_point_weights(state)
        c = float(np.asarray(clearance, dtype=np.float32).reshape(-1)[0])
        mx = 0.0
        for i, leg in enumerate(LEG_ORDER):
            sdf_i = self._numpy_scene_qp_sdf(feet[leg])
            g = float(w[i]) * c - sdf_i
            mx = max(mx, max(0.0, g))
        return float(mx)

    def jax_cfs_alm_g_plus_from_state(self, state: Any, clearance: Any) -> Any:
        if jnp is None:
            raise RuntimeError("jax_cfs_alm_g_plus_from_state requires JAX.")
        feet4 = self.jax_safety_points(state)
        w = self.jax_cfs_safety_point_weights(state)
        c = jnp.asarray(clearance, dtype=jnp.float32).reshape(-1)[0]
        acc = []
        for i in range(4):
            sdf_i = self._jax_scene_qp_sdf(feet4[i])
            gi = w[i] * c - sdf_i
            acc.append(jnp.maximum(jnp.asarray(0.0, dtype=jnp.float32), gi))
        return jnp.max(jnp.stack(acc, axis=0))

    def jax_cfs_safety_point_weights(self, state: Any) -> Any:
        if jnp is None:
            raise RuntimeError("jax_cfs_safety_point_weights requires JAX.")
        x = jnp.asarray(state, dtype=jnp.float32).reshape(-1)[: self.state_dim]
        mode = jnp.mod(jnp.rint(x[14]).astype(jnp.int32), NUM_MODES)
        tau = jnp.clip(x[15], 0.0, 1.0)
        is_m0 = mode == MODE_SUPPORT_FL_RR
        swing_m = jnp.where(
            is_m0,
            jnp.array([0.0, 1.0, 1.0, 0.0], dtype=jnp.float32),
            jnp.array([1.0, 0.0, 0.0, 1.0], dtype=jnp.float32),
        )
        pt_split = jnp.asarray(self.phase_tau_split, dtype=jnp.float32)
        swing_late = jnp.asarray(self.phase_swing_scale_late, dtype=jnp.float32)
        denom = jnp.maximum(jnp.asarray(1e-6, dtype=jnp.float32), jnp.asarray(1.0, dtype=jnp.float32) - pt_split)
        swing_clearance_scale = jnp.where(
            tau < pt_split,
            swing_late,
            swing_late + (jnp.asarray(1.0, dtype=jnp.float32) - swing_late) * (tau - pt_split) / denom,
        )
        return (jnp.asarray(1.0, dtype=jnp.float32) - swing_m) + swing_m * swing_clearance_scale

    def jax_cfs_custom_safety_constraints(
        self,
        state_prev: Any,
        state_t: Any,
        action_ref: Any,
        pts_t: Any,
        J_pts_t: Any,
        clearance_s: Any,
        act_dim: int,
        k_select: int,
    ) -> Any:
        """
        Task-specific foothold-guided constraints for stepping stones.
        Keeps CFSQP solver logic unchanged, but changes only the local constraint geometry.
        """
        if jnp is None:
            raise RuntimeError("jax_cfs_custom_safety_constraints requires JAX.")
        _ = state_prev
        if int(k_select) <= 0:
            return (
                jnp.zeros((0, int(act_dim)), dtype=jnp.float32),
                jnp.zeros((0,), dtype=jnp.float32),
                jnp.zeros((0,), dtype=jnp.bool_),
            )

        x = jnp.asarray(state_t, dtype=jnp.float32).reshape(-1)[: self.state_dim]
        mode = jnp.mod(jnp.rint(x[14]).astype(jnp.int32), NUM_MODES)
        is_m0 = mode == MODE_SUPPORT_FL_RR
        swing_mask = jnp.where(
            is_m0,
            jnp.array([0.0, 1.0, 1.0, 0.0], dtype=jnp.float32),
            jnp.array([1.0, 0.0, 0.0, 1.0], dtype=jnp.float32),
        )
        side_mask = jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float32)
        w_pts = jnp.asarray(self.jax_cfs_safety_point_weights(state_t), dtype=jnp.float32).reshape(-1)[:4]
        c_val = jnp.asarray(clearance_s, dtype=jnp.float32)
        u_ref = jnp.asarray(action_ref, dtype=jnp.float32).reshape(-1)[: int(act_dim)]

        foothold_centers = jnp.asarray(self._foothold_centers_np, dtype=jnp.float32)
        foothold_radii = jnp.asarray(self._foothold_radii_np, dtype=jnp.float32)
        foothold_lane = jnp.asarray(self._foothold_lane_np, dtype=jnp.float32)

        pts4 = jnp.asarray(pts_t, dtype=jnp.float32).reshape(4, 2)
        J4 = jnp.asarray(J_pts_t, dtype=jnp.float32)[:, :, : int(act_dim)]  # (4,2,act_dim)
        body_x = x[0]
        c_scalar = jnp.asarray(clearance_s, dtype=jnp.float32).reshape(-1)[0]
        policy = str(getattr(self, "cfs_custom_constraint_target_policy", "legacy"))

        def _constraint_from_idx(idx, chosen_valid, p, J, wi):
            center = foothold_centers[idx]
            radius = foothold_radii[idx]
            v = center - p
            n = jnp.linalg.norm(v)
            n_safe = jnp.maximum(n, jnp.asarray(1e-6, dtype=jnp.float32))
            grad = v / n_safe
            d_margin = radius - n
            grad_use = jnp.where(chosen_valid, grad, jnp.zeros((2,), dtype=jnp.float32))
            d_use = jnp.where(chosen_valid, d_margin, jnp.asarray(-1e6, dtype=jnp.float32))
            A_row = jnp.matmul(jnp.transpose(J), grad_use)
            b_val = wi * c_scalar - d_use + jnp.dot(A_row, u_ref)
            b_val = jnp.where(chosen_valid, b_val, -jnp.inf)
            A_row = jnp.where(chosen_valid, A_row, jnp.zeros((int(act_dim),), dtype=jnp.float32))
            return A_row, b_val, chosen_valid

        def _one_leg_legacy(i):
            p = pts4[i]
            J = J4[i]
            side = side_mask[i]
            is_swing = swing_mask[i] > 0.5
            wi = w_pts[i]

            dvec = foothold_centers - p[None, :]
            dist = jnp.linalg.norm(dvec, axis=-1)
            lane_ok = jnp.logical_or(jnp.abs(foothold_lane) < 0.5, jnp.sign(foothold_lane) == jnp.sign(side))
            forward_ok = foothold_centers[:, 0] >= (body_x - jnp.asarray(0.02, dtype=jnp.float32))
            valid = jnp.logical_and(lane_ok, jnp.logical_or(jnp.logical_not(is_swing), forward_ok))
            score = dist + jnp.where(valid, 0.0, jnp.asarray(1e3, dtype=jnp.float32))
            idx = jnp.argmin(score)
            chosen_valid = score[idx] < jnp.asarray(900.0, dtype=jnp.float32)
            return _constraint_from_idx(idx, chosen_valid, p, J, wi)

        swing_cap = jnp.asarray(float(self.cfs_custom_swing_forward_cap), dtype=jnp.float32)

        def _one_leg_reachable(i):
            p = pts4[i]
            J = J4[i]
            side = side_mask[i]
            is_swing = swing_mask[i] > 0.5
            wi = w_pts[i]

            dvec = foothold_centers - p[None, :]
            dist = jnp.linalg.norm(dvec, axis=-1)
            lane_ok = jnp.logical_or(jnp.abs(foothold_lane) < 0.5, jnp.sign(foothold_lane) == jnp.sign(side))
            forward_ok = foothold_centers[:, 0] >= (body_x - jnp.asarray(0.02, dtype=jnp.float32))
            reach_ok = foothold_centers[:, 0] <= (p[0] + swing_cap)
            valid_swing_rw = jnp.logical_and(lane_ok, jnp.logical_and(forward_ok, reach_ok))
            valid_support = lane_ok
            valid_legacy = jnp.logical_and(lane_ok, jnp.logical_or(jnp.logical_not(is_swing), forward_ok))

            need_clear = wi * c_scalar
            margin_disk = foothold_radii - dist
            # Support feet: keep nearest-lane stone (same as legacy) so the gradient stays local.
            score_sup = dist + jnp.where(valid_support, 0.0, jnp.asarray(1e3, dtype=jnp.float32))
            idx_sup = jnp.argmin(score_sup)
            ok_sup = score_sup[idx_sup] < jnp.asarray(900.0, dtype=jnp.float32)

            cand_in = jnp.logical_and(valid_swing_rw, margin_disk >= (need_clear - jnp.asarray(1e-3, dtype=jnp.float32)))
            score_in = jnp.where(cand_in, dist, jnp.asarray(jnp.inf, dtype=jnp.float32))
            idx_in = jnp.argmin(score_in)
            has_in = jnp.min(score_in) < jnp.asarray(1.0e6, dtype=jnp.float32)

            margin_masked = jnp.where(valid_swing_rw, margin_disk, jnp.asarray(-1.0e9, dtype=jnp.float32))
            idx_fb = jnp.argmax(margin_masked)
            has_rw = jnp.max(margin_masked) > jnp.asarray(-1.0e8, dtype=jnp.float32)

            score_lg = dist + jnp.where(valid_legacy, 0.0, jnp.asarray(1e3, dtype=jnp.float32))
            idx_lg = jnp.argmin(score_lg)

            idx_sw = jnp.where(has_in, idx_in, jnp.where(has_rw, idx_fb, idx_lg))
            ok_sw_in = jnp.logical_or(
                has_in,
                jnp.logical_or(
                    has_rw,
                    score_lg[idx_lg] < jnp.asarray(900.0, dtype=jnp.float32),
                ),
            )
            idx = jnp.where(is_swing, idx_sw, idx_sup)
            chosen_valid = jnp.where(is_swing, ok_sw_in, ok_sup)
            return _constraint_from_idx(idx, chosen_valid, p, J, wi)

        _one_leg = _one_leg_reachable if policy == "reachable" else _one_leg_legacy
        A4, b4, v4 = jax.vmap(_one_leg)(jnp.arange(4, dtype=jnp.int32))

        K = int(k_select)
        A_full = jnp.zeros((K, int(act_dim)), dtype=jnp.float32)
        b_full = jnp.full((K,), -jnp.inf, dtype=jnp.float32)
        v_full = jnp.zeros((K,), dtype=jnp.bool_)
        use_n = min(4, K)
        A_full = A_full.at[:use_n, :].set(A4[:use_n, :])
        b_full = b_full.at[:use_n].set(b4[:use_n])
        v_full = v_full.at[:use_n].set(v4[:use_n])
        return A_full, b_full, v_full


def make_stepping_stones_energy(env: QuadrupedSteppingStones2DEnv) -> LegacyEnergyFunctional:
    scene = env.scene
    centers = np.asarray(scene.stones_centers, dtype=np.float32)
    radii = np.asarray(scene.stones_radii, dtype=np.float32)
    platforms = np.asarray(getattr(scene, "support_platforms", np.zeros((0, 4))), dtype=np.float32)
    n_sp = int(platforms.shape[0])
    plat_pad_np = np.zeros((4, 4), dtype=np.float32)
    if n_sp > 0:
        plat_pad_np[:n_sp] = platforms[:4]
    goal_mid = np.asarray(scene.goal_mid, dtype=np.float32)
    start_mid = np.asarray(scene.start_mid, dtype=np.float32)
    river_x = np.asarray(scene.river_x, dtype=np.float32)
    has_river = bool(getattr(scene, "has_river", True))
    cy = float(env.centerline_y)
    margin = float(env.stone_margin)
    H_plan = int(getattr(env, "horizon", 12))

    def _body_xy_np(s: float, ey: float) -> np.ndarray:
        return np.asarray([float(s), cy + float(ey)], dtype=np.float32)

    def _support_violation_np(p: np.ndarray) -> float:
        stone_v = foot_stepping_violation_np(p, centers, radii, platforms, margin)
        if has_river:
            p = np.asarray(p, dtype=np.float32).reshape(2)
            if float(p[0]) <= float(river_x[0]) or float(p[0]) >= float(river_x[1]):
                return 0.0
            bank_v = min(abs(float(p[0]) - float(river_x[0])), abs(float(p[0]) - float(river_x[1])))
            return min(stone_v, bank_v)
        return stone_v

    def _support_center_np(feet: Dict[str, np.ndarray], mode: int) -> np.ndarray:
        if int(mode) == MODE_SUPPORT_FL_RR:
            return (0.5 * (feet["FL"] + feet["RR"])).astype(np.float32)
        return (0.5 * (feet["FR"] + feet["RL"])).astype(np.float32)

    nominal_xf = float(env.x_f_nominal)
    nominal_xr = float(env.x_r_nominal)
    nominal_yL = float(env.y_L_nominal)
    nominal_yR = float(env.y_R_nominal)
    w_term_foot = float(getattr(env, "task_terminal_foot_weight", 0.55))

    if jnp is not None:
        centers_j = jnp.asarray(centers, dtype=jnp.float32)
        radii_j = jnp.asarray(radii, dtype=jnp.float32)
        margin_j = jnp.asarray(margin, dtype=jnp.float32)
        plat_pad_j = jnp.asarray(plat_pad_np, dtype=jnp.float32)
        n_sp_j = int(n_sp)
        goal_mid_j = jnp.asarray(goal_mid, dtype=jnp.float32)
        start_mid_j = jnp.asarray(start_mid, dtype=jnp.float32)
        river_x_j = jnp.asarray(river_x, dtype=jnp.float32)
        has_river_j = jnp.asarray(1.0 if has_river else 0.0, dtype=jnp.float32)
        cy_j = jnp.asarray(cy, dtype=jnp.float32)
        Hj = float(H_plan)
        w_tf = w_term_foot

        def _decode_j(x):
            x = jnp.asarray(x, dtype=jnp.float32).reshape(-1)
            s, ey, epsi = x[0], x[1], x[2]
            vel = x[3:6]
            res = x[6:14].reshape(4, 2)
            mode = jnp.mod(jnp.rint(x[14]).astype(jnp.int32), NUM_MODES)
            tau = jnp.clip(x[15], 0.0, 1.0)
            body = jnp.stack([s, cy_j + ey])
            return body, epsi, vel, res, mode, tau

        def _feet_j(s, ey, epsi, res):
            body = jnp.stack([s, cy_j + ey], dtype=jnp.float32)
            c = jnp.cos(epsi)
            s_ = jnp.sin(epsi)
            R = jnp.asarray([[c, -s_], [s_, c]], dtype=jnp.float32)
            tpl = jnp.stack(
                [
                    jnp.stack([nominal_xf, nominal_yL]),
                    jnp.stack([nominal_xf, nominal_yR]),
                    jnp.stack([nominal_xr, nominal_yL]),
                    jnp.stack([nominal_xr, nominal_yR]),
                ],
                axis=0,
            )
            local = tpl + res
            return body[None, :] + local @ R.T

        def _support_violation_j(p):
            if centers.shape[0] > 0:
                d = jnp.linalg.norm(centers_j - p[None, :], axis=-1)
                vd = jnp.min(jnp.maximum(d - (radii_j - margin_j), 0.0))
            else:
                vd = jnp.asarray(1e9, dtype=jnp.float32)

            def pb(bounds):
                xm, xM, ym, yM = bounds[0], bounds[1], bounds[2], bounds[3]
                xl = xm + margin_j
                xR = xM - margin_j
                yb = ym + margin_j
                yT = yM - margin_j
                ins = (p[0] >= xl) & (p[0] <= xR) & (p[1] >= yb) & (p[1] <= yT)
                dx = jnp.maximum(0.0, xl - p[0]) + jnp.maximum(0.0, p[0] - xR)
                dy = jnp.maximum(0.0, yb - p[1]) + jnp.maximum(0.0, p[1] - yT)
                dist = jnp.sqrt(dx * dx + dy * dy)
                return jnp.where(ins, 0.0, dist)

            p0, p1, p2, p3 = pb(plat_pad_j[0]), pb(plat_pad_j[1]), pb(plat_pad_j[2]), pb(plat_pad_j[3])
            stacked = jnp.stack([p0, p1, p2, p3])
            msk = jnp.array([1.0 if k < n_sp_j else 0.0 for k in range(4)], dtype=jnp.float32)
            vp = jnp.min(jnp.where(msk > 0.5, stacked, jnp.asarray(1e9, dtype=jnp.float32)))
            stone_v = jnp.minimum(vd, vp)
            in_bank = jnp.logical_or(p[0] <= river_x_j[0], p[0] >= river_x_j[1])
            bank_v = jnp.minimum(jnp.abs(p[0] - river_x_j[0]), jnp.abs(p[0] - river_x_j[1]))
            river_v = jnp.where(in_bank, 0.0, jnp.minimum(stone_v, bank_v))
            return jnp.where(has_river_j > 0.5, river_v, stone_v)

        def task_energy(x, u, ctx):
            _ = u
            body, psi, vel, _res, mode, _tau = _decode_j(x)
            feet4 = _feet_j(body[0], body[1] - cy_j, psi, _res)
            feet = {"FL": feet4[0], "FR": feet4[1], "RL": feet4[2], "RR": feet4[3]}
            goal_vec = goal_mid_j - body
            psi_ref = jnp.arctan2(goal_vec[1], goal_vec[0])
            h_err = jnp.arctan2(jnp.sin(psi - psi_ref), jnp.cos(psi - psi_ref))
            support_center = jnp.where(
                mode == MODE_SUPPORT_FL_RR,
                0.5 * (feet["FL"] + feet["RR"]),
                0.5 * (feet["FR"] + feet["RL"]),
            )
            c_g = jnp.cos(psi_ref)
            s_g = jnp.sin(psi_ref)
            Rg = jnp.asarray([[c_g, -s_g], [s_g, c_g]], dtype=jnp.float32)
            tplg = jnp.stack(
                [
                    jnp.stack([nominal_xf, nominal_yL]),
                    jnp.stack([nominal_xf, nominal_yR]),
                    jnp.stack([nominal_xr, nominal_yL]),
                    jnp.stack([nominal_xr, nominal_yR]),
                ],
                axis=0,
            )
            gf = goal_mid_j[None, :] + tplg @ Rg.T
            foot_err = sum(jnp.sum((feet[LEG_ORDER[k]] - gf[k]) ** 2) for k in range(4))
            info = ctx or {}
            t = info.get("t", 0.0)
            tscalar = jnp.asarray(t, dtype=jnp.float32).reshape(-1)[0]
            # Phase-dependent foot tracking: ramp weight over the horizon
            # so swing feet get directional guidance throughout.
            progress = jnp.clip(tscalar / jnp.maximum(Hj - 1.0, 1.0), 0.0, 1.0)
            w_foot = 0.02 + (w_tf - 0.02) * (progress ** 2)
            vel_err = (
                1.20 * ((body[1] - goal_mid_j[1]) ** 2)
                + 0.80 * (vel[1] ** 2)
                + 0.50 * (vel[2] ** 2)
                + 0.25 * (jnp.maximum(0.0, -vel[0]) ** 2)
            )
            pr_cost = jnp.maximum(0.0, -(body[0] - goal_mid_j[0])) * 0.15

            # Time-smooth progress guidance (à la MDOC):
            # Penalize deviation from expected linear distance decay.
            d0_j = jnp.linalg.norm(start_mid_j - goal_mid_j)
            dist_to_goal = jnp.linalg.norm(body - goal_mid_j)
            expected_dist = d0_j * jnp.maximum(0.0, 1.0 - progress)
            guide_cost = 0.8 * (dist_to_goal - expected_dist) ** 2

            return (
                jnp.sum((body - goal_mid_j) ** 2)
                + pr_cost
                + guide_cost
                + vel_err
                + 0.08 * (h_err**2)
                + 1.05 * jnp.sum((body - support_center) ** 2)
                + w_foot * foot_err
            )

        def smooth_energy(x, u, ctx):
            _ = (x, ctx)
            u = jnp.asarray(u, dtype=jnp.float32).reshape(-1)
            return (
                0.165 * jnp.sum(u[:2] ** 2)
                + 0.15 * (u[2] ** 2)
                + 0.165 * jnp.sum(u[3:11] ** 2)
                + 0.042 * (u[11] ** 2)
            )

        def foothold_energy(x, u, ctx):
            _ = (u, ctx)
            body, psi, _vel, res, _mode, _tau = _decode_j(x)
            feet4 = _feet_j(body[0], body[1] - cy_j, psi, res)
            feet = {"FL": feet4[0], "FR": feet4[1], "RL": feet4[2], "RR": feet4[3]}
            return 40.0 * sum(_support_violation_j(feet[leg]) for leg in LEG_ORDER)

        def support_energy(x, u, ctx):
            _ = (u, ctx)
            body, psi, _vel, res, mode, _tau = _decode_j(x)
            feet4 = _feet_j(body[0], body[1] - cy_j, psi, res)
            feet = {"FL": feet4[0], "FR": feet4[1], "RL": feet4[2], "RR": feet4[3]}
            center = jnp.where(
                mode == MODE_SUPPORT_FL_RR,
                0.5 * (feet["FL"] + feet["RR"]),
                0.5 * (feet["FR"] + feet["RL"]),
            )
            return 14.0 * jnp.sum((body - center) ** 2)

        def geometry_energy(x, u, ctx):
            _ = (u, ctx)
            body, psi, _vel, res, _mode, _tau = _decode_j(x)
            c = jnp.cos(psi)
            s = jnp.sin(psi)
            Rinv = jnp.asarray([[c, s], [-s, c]], dtype=jnp.float32)
            tpl = jnp.stack(
                [
                    jnp.stack([nominal_xf, nominal_yL]),
                    jnp.stack([nominal_xf, nominal_yR]),
                    jnp.stack([nominal_xr, nominal_yL]),
                    jnp.stack([nominal_xr, nominal_yR]),
                ],
                axis=0,
            )
            nominal_tpl = tpl
            feet4 = body[None, :] + (tpl + res) @ jnp.asarray([[c, -s], [s, c]], dtype=jnp.float32).T
            shape = 0.0
            for k in range(4):
                local = Rinv @ (feet4[k] - body)
                shape = shape + jnp.sum((local - nominal_tpl[k]) ** 2)
            res_reg = jnp.sum(res**2)
            return 1.2 * shape + 0.8 * res_reg

        def step_bound_energy(x, u, ctx):
            _ = (x, ctx)
            u = jnp.asarray(u, dtype=jnp.float32).reshape(-1)
            body_ex = jnp.maximum(0.0, jnp.linalg.norm(u[:2]) - env.body_step_norm_max)
            yaw_v = jnp.maximum(0.0, jnp.abs(u[2]) - env.omega_max)
            dres = u[3:11].reshape(4, 2)
            dres_ex = jnp.maximum(0.0, jnp.max(jnp.linalg.norm(dres, axis=-1)) - env.residual_rate_limit)
            dtau_v = jnp.maximum(0.0, u[11] - env.phase_rate_limit)
            dtau_neg = jnp.maximum(0.0, -u[11])
            return 8.0 * (body_ex**2 + dres_ex**2) + 4.0 * (yaw_v**2 + dtau_v**2 + dtau_neg**2)

    else:

        def task_energy(x, u, ctx):
            _ = u
            x = np.asarray(x, dtype=np.float32).reshape(-1)
            s, ey, epsi, vel, res, mode, _tau = env._decode_state(x)
            body = _body_xy_np(s, ey)
            feet = env._decode_feet_np(s, ey, epsi, res)
            goal_vec = goal_mid - body
            psi_ref = float(np.arctan2(float(goal_vec[1]), float(goal_vec[0])))
            h_err = _wrap_np(float(epsi) - psi_ref)
            support_center = _support_center_np(feet, mode)
            goal_feet = env._nominal_feet(goal_mid, psi_ref)
            foot_err = sum(float(np.sum((feet[leg] - goal_feet[leg]) ** 2)) for leg in LEG_ORDER)
            info = ctx or {}
            t = info.get("t", 0.0)
            try:
                tv = float(np.asarray(t).reshape(-1)[0])
            except Exception:
                tv = 0.0
            is_term = tv >= float(H_plan) - 1.5
            w_foot = float(w_term_foot) if is_term else 0.02
            vel_err = (
                1.20 * float((body[1] - goal_mid[1]) ** 2)
                + 0.80 * float(vel[1] ** 2)
                + 0.50 * float(vel[2] ** 2)
                + 0.25 * max(0.0, -float(vel[0])) ** 2
            )
            pr_cost = max(0.0, -float(body[0] - goal_mid[0])) * 0.15
            return float(
                np.sum((body - goal_mid) ** 2)
                + pr_cost
                + vel_err
                + 0.08 * (h_err**2)
                + 1.05 * np.sum((body - support_center) ** 2)
                + w_foot * foot_err
            )

        def smooth_energy(x, u, ctx):
            _ = (x, ctx)
            u = np.asarray(u, dtype=np.float32).reshape(-1)
            return float(
                0.165 * np.sum(u[:2] ** 2)
                + 0.15 * float(u[2] ** 2)
                + 0.165 * np.sum(u[3:11] ** 2)
                + 0.042 * float(u[11] ** 2)
            )

        def foothold_energy(x, u, ctx):
            _ = (u, ctx)
            feet = env._feet_from_state_np(np.asarray(x, dtype=np.float32).reshape(-1))
            return 26.0 * sum(_support_violation_np(feet[leg]) for leg in LEG_ORDER)

        def support_energy(x, u, ctx):
            _ = (u, ctx)
            x = np.asarray(x, dtype=np.float32).reshape(-1)
            s, ey, epsi, vel, res, mode, _tau = env._decode_state(x)
            body = _body_xy_np(s, ey)
            feet = env._decode_feet_np(s, ey, epsi, res)
            center = _support_center_np(feet, mode)
            return 14.0 * float(np.sum((body - center) ** 2))

        def geometry_energy(x, u, ctx):
            _ = (u, ctx)
            x = np.asarray(x, dtype=np.float32).reshape(-1)
            s, ey, epsi, _vel, res, _mode, _tau = env._decode_state(x)
            body = _body_xy_np(s, ey)
            feet4 = env._decode_feet_np(s, ey, epsi, res)
            c, s = float(np.cos(epsi)), float(np.sin(epsi))
            Rinv = np.asarray([[c, s], [-s, c]], dtype=np.float32)
            tpl = env._template_local_np()
            nom = tpl
            shape = 0.0
            for k, leg in enumerate(LEG_ORDER):
                local = Rinv @ (feet4[leg] - body)
                shape = shape + float(np.sum((local - nom[leg]) ** 2))
            res_reg = float(np.sum(res**2))
            return float(1.2 * shape + 0.8 * res_reg)

        def step_bound_energy(x, u, ctx):
            _ = (x, ctx)
            u = np.asarray(u, dtype=np.float32).reshape(-1)
            body_ex = max(0.0, float(np.linalg.norm(u[:2])) - env.body_step_norm_max)
            yaw_v = max(0.0, abs(float(u[2])) - env.omega_max)
            dres = u[3:11].reshape(4, 2)
            dres_ex = max(0.0, float(np.max(np.linalg.norm(dres, axis=-1))) - env.residual_rate_limit)
            dtau_v = max(0.0, float(u[11]) - env.phase_rate_limit)
            dtau_neg = max(0.0, -float(u[11]))
            return float(8.0 * (body_ex**2 + dres_ex**2) + 4.0 * (yaw_v**2 + dtau_v**2 + dtau_neg**2))

    return LegacyEnergyFunctional(
        {
            "task": EnergyTerm(task_energy, 1.0),
            "smooth": EnergyTerm(smooth_energy, 0.92),
            "foothold": EnergyTerm(foothold_energy, 1.12),
            "support": EnergyTerm(support_energy, 1.08),
            "geometry": EnergyTerm(geometry_energy, 1.0),
            "step_bound": EnergyTerm(step_bound_energy, 1.0),
        }
    )
