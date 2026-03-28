from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import jax.numpy as jnp
    import jax.lax as lax
except Exception:
    jnp = None
    lax = None

from genedynamics.core.energy import EnergyTerm, LegacyEnergyFunctional
from genedynamics.tasks.stepping_stones import SteppingStonesScene, sample_stepping_stones_scene

MODE_DS_FL_RR = 0
MODE_QS_AFTER_FL_RR = 1
MODE_DS_FR_RL = 2
MODE_QS_AFTER_FR_RL = 3
NUM_MODES = 4


def _to_np(x: Any, n: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size < n:
        arr = np.pad(arr, (0, n - arr.size), constant_values=0.0)
    return arr[:n]


def _wrap_np(theta: float) -> float:
    return float(np.arctan2(np.sin(theta), np.cos(theta)))


def _rotation_np(yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.asarray([[c, -s], [s, c]], dtype=np.float32)


def _clip_norm_np(v: np.ndarray, limit: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= limit or n < 1e-8:
        return v.astype(np.float32)
    return ((float(limit) / n) * v).astype(np.float32)


@dataclass
class QuadrupedSteppingStones2DEnv:
    """
    Hybrid reduced-order quadruped stepping planner.

    State  : [b_x, b_y, psi, v_x, v_y, omega, x_L, x_R, l_L, l_R, mode, tau]
    Action : [a_x, a_y, alpha, dl_L, dl_R, dtau]

    Feet are reconstructed analytically from left/right anchors, fore-hind lengths
    and contact mode, so the planner never optimizes four independent foot xy states.
    """

    dt: float = 1.0
    horizon: int = 12
    control_limit: float = 0.45
    l_max: float = 0.35
    body_shift_limit: Optional[float] = None
    stance_width: float = 0.30
    step_width: Optional[float] = None
    fore_hind_offset: float = 0.18
    yaw_step_limit: float = 0.45
    terminal_foot_penalty: float = 0.35
    terminal_foot_margin: float = 0.20
    terminal_reward_weight: float = 300.0
    start_mid: Tuple[float, float] = (-1.25, 0.0)
    goal_mid: Tuple[float, float] = (1.25, 0.0)
    obstacles: Optional[Any] = None
    scene_seed: int = 0
    scene_level: int = 1
    scene: Optional[SteppingStonesScene] = None

    act_dim: int = 6
    state_dim: int = 12
    enable_cfs_safety_points_qp: bool = True

    def __post_init__(self) -> None:
        scene = self.scene
        if scene is None and self.obstacles is not None and hasattr(self.obstacles, "stepping_scene"):
            scene = getattr(self.obstacles, "stepping_scene")
        if scene is None:
            scene = sample_stepping_stones_scene(
                level=int(self.scene_level),
                seed=int(self.scene_seed),
                l_max=float(self.l_max),
                stance_width=float(self.stance_width),
                start_mid=self.start_mid,
                goal_mid=self.goal_mid,
                fore_hind_offset=float(self.fore_hind_offset),
            )
        self.scene = scene
        self.horizon = max(int(self.horizon), int(scene.k_horizon))
        self.l_max = float(scene.l_max)
        self.step_width = float(self.step_width if self.step_width is not None else self.stance_width)
        self.body_shift_limit = float(
            self.body_shift_limit if self.body_shift_limit is not None else min(0.26, 0.80 * self.l_max)
        )
        self.body_speed_limit = float(self.body_shift_limit)
        self.body_speed_limit_x = float(self.body_speed_limit)
        self.body_speed_limit_y = float(min(0.08, 0.40 * self.body_speed_limit))
        self.body_y_limit = float(min(0.15, 0.50 * self.step_width))
        self.body_acc_limit = float(self.control_limit)
        self.length_nominal = float(2.0 * self.fore_hind_offset)
        self.length_rate_limit = float(max(0.10, min(self.l_max, self.control_limit)))
        self.length_min = float(max(0.10, self.length_nominal - self.l_max))
        self.length_max = float(self.length_nominal + self.l_max)
        self.phase_rate_limit = float(max(0.15, min(0.50, self.control_limit)))
        self.yaw_acc_limit = float(self.control_limit)
        self.yaw_rate_limit = float(min(0.20, self.yaw_step_limit))
        self.control_limit = max(
            float(self.control_limit),
            float(self.body_acc_limit),
            float(self.length_rate_limit),
            float(self.phase_rate_limit),
        )
        self.target = np.asarray(scene.goal_mid, dtype=np.float32)
        centers = np.asarray(scene.stones_centers, dtype=np.float32)
        left_mask = centers[:, 1] >= float(self.target[1])
        right_mask = ~left_mask
        self.left_lane_xs = np.sort(centers[left_mask, 0].astype(np.float32))
        self.right_lane_xs = np.sort(centers[right_mask, 0].astype(np.float32))
        start_mid = np.asarray(scene.start_mid, dtype=np.float32)
        goal_mid = np.asarray(scene.goal_mid, dtype=np.float32)
        psi0 = float(np.arctan2(goal_mid[1] - start_mid[1], goal_mid[0] - start_mid[0]))
        x_l0 = float(start_mid[0] + self.fore_hind_offset)
        x_r0 = float(start_mid[0] - self.fore_hind_offset)
        self._default_state = self._pack_state(
            body=start_mid,
            psi=psi0,
            vel=np.zeros((3,), dtype=np.float32),
            x_l=x_l0,
            x_r=x_r0,
            l_l=self.length_nominal,
            l_r=self.length_nominal,
            mode=MODE_DS_FL_RR,
            tau=0.0,
        )

    def _nominal_offsets(self) -> Dict[str, np.ndarray]:
        half_w = 0.5 * float(self.step_width)
        fh = float(self.fore_hind_offset)
        return {
            "FL": np.asarray([fh, half_w], dtype=np.float32),
            "FR": np.asarray([fh, -half_w], dtype=np.float32),
            "RL": np.asarray([-fh, half_w], dtype=np.float32),
            "RR": np.asarray([-fh, -half_w], dtype=np.float32),
        }

    def _nominal_feet(self, body: np.ndarray, psi: float) -> Dict[str, np.ndarray]:
        R = _rotation_np(psi)
        feet = {}
        for leg, off in self._nominal_offsets().items():
            feet[leg] = (np.asarray(body, dtype=np.float32) + R @ off).astype(np.float32)
        return feet

    def _feet_from_components(
        self,
        x_l: np.ndarray | float,
        x_r: np.ndarray | float,
        l_l: np.ndarray | float,
        l_r: np.ndarray | float,
        mode: np.ndarray | int,
    ) -> Dict[str, np.ndarray]:
        x_l = np.asarray(x_l, dtype=np.float32)
        x_r = np.asarray(x_r, dtype=np.float32)
        l_l = np.asarray(l_l, dtype=np.float32)
        l_r = np.asarray(l_r, dtype=np.float32)
        mode = np.asarray(mode, dtype=np.int32)
        half_w = 0.5 * float(self.step_width)
        left_y = np.broadcast_to(np.asarray(half_w, dtype=np.float32), x_l.shape)
        right_y = np.broadcast_to(np.asarray(-half_w, dtype=np.float32), x_r.shape)

        fl = np.stack([x_l, left_y], axis=-1)
        fr = np.stack([x_r, right_y], axis=-1)
        rl = np.stack([x_l, left_y], axis=-1)
        rr = np.stack([x_r, right_y], axis=-1)

        mask_flrr = np.logical_or(mode == MODE_DS_FL_RR, mode == MODE_QS_AFTER_FL_RR)
        mask_frrl = np.logical_or(mode == MODE_DS_FR_RL, mode == MODE_QS_AFTER_FR_RL)

        fl[..., 0] = np.where(mask_flrr, x_l, x_l + l_l)
        rl[..., 0] = np.where(mask_flrr, x_l - l_l, x_l)
        rr[..., 0] = np.where(mask_flrr, x_r, x_r - l_r)
        fr[..., 0] = np.where(mask_flrr, x_r + l_r, x_r)
        return {"FL": fl.astype(np.float32), "FR": fr.astype(np.float32), "RL": rl.astype(np.float32), "RR": rr.astype(np.float32)}

    def _pack_state(
        self,
        *,
        body: np.ndarray,
        psi: float,
        vel: np.ndarray,
        x_l: float,
        x_r: float,
        l_l: float,
        l_r: float,
        mode: int,
        tau: float,
    ) -> np.ndarray:
        x = np.zeros((self.state_dim,), dtype=np.float32)
        x[:2] = np.asarray(body, dtype=np.float32)
        x[2] = float(psi)
        x[3:6] = np.asarray(vel, dtype=np.float32)
        x[6] = float(x_l)
        x[7] = float(x_r)
        x[8] = float(l_l)
        x[9] = float(l_r)
        x[10] = float(int(mode) % NUM_MODES)
        x[11] = float(np.clip(tau, 0.0, 1.0))
        return x

    def _decode_state(self, state: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, float, float, float, int, float]:
        x = _to_np(state, self.state_dim)
        body = x[:2].astype(np.float32)
        psi = float(x[2])
        vel = x[3:6].astype(np.float32)
        x_l = float(x[6])
        x_r = float(x[7])
        l_l = float(x[8])
        l_r = float(x[9])
        mode = int(np.rint(float(x[10]))) % NUM_MODES
        tau = float(np.clip(x[11], 0.0, 1.0))
        return body, psi, vel, x_l, x_r, l_l, l_r, mode, tau

    def _snap_lane_x_np(self, x: float, side: str) -> float:
        lane_xs = self.left_lane_xs if side == "left" else self.right_lane_xs
        if lane_xs.size == 0:
            return float(x)
        idx = int(np.argmin(np.abs(lane_xs - float(x))))
        return float(lane_xs[idx])

    def _mode_support_center_np(self, feet: Dict[str, np.ndarray], mode: int) -> np.ndarray:
        if mode == MODE_DS_FL_RR:
            return 0.5 * (feet["FL"] + feet["RR"])
        if mode == MODE_DS_FR_RL:
            return 0.5 * (feet["FR"] + feet["RL"])
        return 0.25 * (feet["FL"] + feet["FR"] + feet["RL"] + feet["RR"])

    def _reconstruct_body_from_components_np(
        self,
        x_l: float,
        x_r: float,
        l_l: float,
        l_r: float,
        mode: int,
    ) -> np.ndarray:
        feet = self._feet_from_components(x_l, x_r, l_l, l_r, mode)
        body = self._mode_support_center_np(feet, int(mode))
        return np.asarray(body, dtype=np.float32)

    def _apply_mode_reset_np(self, x_l: float, x_r: float, l_l: float, l_r: float, mode: int, tau: float) -> Tuple[float, float, int, float]:
        new_mode = int(mode)
        new_x_l = float(x_l)
        new_x_r = float(x_r)
        if tau < 1.0:
            return new_x_l, new_x_r, new_mode, tau
        if mode == MODE_DS_FL_RR:
            new_mode = MODE_QS_AFTER_FL_RR
        elif mode == MODE_QS_AFTER_FL_RR:
            new_x_l = self._snap_lane_x_np(float(x_l - l_l), "left")
            new_x_r = self._snap_lane_x_np(float(x_r + l_r), "right")
            new_mode = MODE_DS_FR_RL
        elif mode == MODE_DS_FR_RL:
            new_mode = MODE_QS_AFTER_FR_RL
        else:
            new_x_l = self._snap_lane_x_np(float(x_l + l_l), "left")
            new_x_r = self._snap_lane_x_np(float(x_r - l_r), "right")
            new_mode = MODE_DS_FL_RR
        return new_x_l, new_x_r, new_mode, 0.0

    def reset(self, rng: Optional[Any] = None, seed: Optional[int] = None, **kwargs: Any):
        _ = (rng, seed, kwargs)
        return self._default_state.copy(), {}

    def transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        _body, psi, _vel, x_l, x_r, l_l, l_r, mode, tau = self._decode_state(state)
        u = np.clip(_to_np(action, self.act_dim), -self.control_limit, self.control_limit)
        curr_body = self._reconstruct_body_from_components_np(x_l, x_r, l_l, l_r, mode)
        next_omega = float(np.clip(self.dt * float(u[2]), -self.yaw_rate_limit, self.yaw_rate_limit))
        next_psi = _wrap_np(psi + self.dt * next_omega)

        ds_active = 1.0 if mode in (MODE_DS_FL_RR, MODE_DS_FR_RL) else 0.0
        common_dl = 0.5 * float(u[0])
        diff_dl = 0.5 * float(u[1])
        next_l_l = float(np.clip(l_l + ds_active * self.dt * (common_dl + diff_dl + float(u[3])), self.length_min, self.length_max))
        next_l_r = float(np.clip(l_r + ds_active * self.dt * (common_dl - diff_dl + float(u[4])), self.length_min, self.length_max))
        next_tau = float(np.clip(tau + self.dt * float(np.clip(u[5], 0.0, self.phase_rate_limit)), 0.0, 1.0))
        next_x_l, next_x_r, next_mode, next_tau = self._apply_mode_reset_np(x_l, x_r, next_l_l, next_l_r, mode, next_tau)
        next_body = self._reconstruct_body_from_components_np(next_x_l, next_x_r, next_l_l, next_l_r, next_mode)
        next_vel_xy = (next_body - curr_body) / max(float(self.dt), 1e-6)
        next_vel_xy[0] = float(np.clip(next_vel_xy[0], -self.body_speed_limit_x, self.body_speed_limit_x))
        next_vel_xy[1] = float(np.clip(next_vel_xy[1], -self.body_speed_limit_y, self.body_speed_limit_y))
        return self._pack_state(
            body=next_body,
            psi=next_psi,
            vel=np.asarray([next_vel_xy[0], next_vel_xy[1], next_omega], dtype=np.float32),
            x_l=next_x_l,
            x_r=next_x_r,
            l_l=next_l_l,
            l_r=next_l_r,
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

    def jax_transition(self, state: Any, action: Any) -> Any:
        if jnp is None or lax is None:
            raise RuntimeError("jax_transition requires JAX.")
        x = jnp.asarray(state, dtype=jnp.float32).reshape(-1)[: self.state_dim]
        u = jnp.asarray(action, dtype=jnp.float32).reshape(-1)[: self.act_dim]
        u = jnp.clip(u, -self.control_limit, self.control_limit)
        body = x[:2]
        psi = x[2]
        vel_xy = x[3:5]
        omega = x[5]
        x_l = x[6]
        x_r = x[7]
        l_l = x[8]
        l_r = x[9]
        mode = jnp.mod(jnp.rint(x[10]).astype(jnp.int32), NUM_MODES)
        tau = jnp.clip(x[11], 0.0, 1.0)

        def _clip_norm(v, limit):
            n = jnp.linalg.norm(v)
            scale = jnp.where(n > limit, limit / jnp.maximum(n, 1e-8), 1.0)
            return v * scale

        left_lane_xs = jnp.asarray(self.left_lane_xs, dtype=jnp.float32)
        right_lane_xs = jnp.asarray(self.right_lane_xs, dtype=jnp.float32)
        half_w_j = jnp.asarray(0.5 * float(self.step_width), dtype=jnp.float32)

        def _snap_lane_x_j(x_in, lane_xs):
            idx = jnp.argmin(jnp.abs(lane_xs - x_in))
            return lane_xs[idx]

        def _feet_j(xl, xr, ll, lr, mode_in):
            mask = jnp.logical_or(mode_in == MODE_DS_FL_RR, mode_in == MODE_QS_AFTER_FL_RR)
            fl = jnp.stack([jnp.where(mask, xl, xl + ll), half_w_j])
            rl = jnp.stack([jnp.where(mask, xl - ll, xl), half_w_j])
            rr = jnp.stack([jnp.where(mask, xr, xr - lr), -half_w_j])
            fr = jnp.stack([jnp.where(mask, xr + lr, xr), -half_w_j])
            return fl, fr, rl, rr

        def _body_from_components_j(xl, xr, ll, lr, mode_in):
            fl, fr, rl, rr = _feet_j(xl, xr, ll, lr, mode_in)
            center_flrr = 0.5 * (fl + rr)
            center_frrl = 0.5 * (fr + rl)
            center_qs = 0.25 * (fl + fr + rl + rr)
            return jnp.where(
                mode_in == MODE_DS_FL_RR,
                center_flrr,
                jnp.where(mode_in == MODE_DS_FR_RL, center_frrl, center_qs),
            )

        curr_body = _body_from_components_j(x_l, x_r, l_l, l_r, mode)
        next_omega = jnp.clip(self.dt * u[2], -self.yaw_rate_limit, self.yaw_rate_limit)
        next_psi = jnp.arctan2(jnp.sin(psi + self.dt * next_omega), jnp.cos(psi + self.dt * next_omega))
        ds_active = jnp.where(jnp.logical_or(mode == MODE_DS_FL_RR, mode == MODE_DS_FR_RL), 1.0, 0.0)
        common_dl = 0.5 * u[0]
        diff_dl = 0.5 * u[1]
        next_l_l = jnp.clip(l_l + ds_active * self.dt * (common_dl + diff_dl + u[3]), self.length_min, self.length_max)
        next_l_r = jnp.clip(l_r + ds_active * self.dt * (common_dl - diff_dl + u[4]), self.length_min, self.length_max)
        next_tau = jnp.clip(tau + self.dt * jnp.clip(u[5], 0.0, self.phase_rate_limit), 0.0, 1.0)

        def no_reset(vals):
            return vals

        def do_reset(vals):
            xl0, xr0, ll0, lr0, mode0, _tau0 = vals
            xl = xl0
            xr = xr0
            mode_next = mode0
            xl = lax.select(mode0 == MODE_QS_AFTER_FL_RR, _snap_lane_x_j(xl0 - ll0, left_lane_xs), xl)
            xr = lax.select(mode0 == MODE_QS_AFTER_FL_RR, _snap_lane_x_j(xr0 + lr0, right_lane_xs), xr)
            xl = lax.select(mode0 == MODE_QS_AFTER_FR_RL, _snap_lane_x_j(xl0 + ll0, left_lane_xs), xl)
            xr = lax.select(mode0 == MODE_QS_AFTER_FR_RL, _snap_lane_x_j(xr0 - lr0, right_lane_xs), xr)
            mode_next = lax.select(mode0 == MODE_DS_FL_RR, jnp.asarray(MODE_QS_AFTER_FL_RR, dtype=jnp.int32), mode_next)
            mode_next = lax.select(mode0 == MODE_QS_AFTER_FL_RR, jnp.asarray(MODE_DS_FR_RL, dtype=jnp.int32), mode_next)
            mode_next = lax.select(mode0 == MODE_DS_FR_RL, jnp.asarray(MODE_QS_AFTER_FR_RL, dtype=jnp.int32), mode_next)
            mode_next = lax.select(mode0 == MODE_QS_AFTER_FR_RL, jnp.asarray(MODE_DS_FL_RR, dtype=jnp.int32), mode_next)
            return xl, xr, ll0, lr0, mode_next, jnp.asarray(0.0, dtype=jnp.float32)

        next_x_l, next_x_r, next_l_l, next_l_r, next_mode, next_tau = lax.cond(
            next_tau >= 1.0,
            do_reset,
            no_reset,
            (x_l, x_r, next_l_l, next_l_r, mode, next_tau),
        )
        next_body = _body_from_components_j(next_x_l, next_x_r, next_l_l, next_l_r, next_mode)
        next_vel_xy = (next_body - curr_body) / jnp.maximum(jnp.asarray(self.dt, dtype=jnp.float32), 1e-6)
        next_vel_xy = next_vel_xy.at[0].set(jnp.clip(next_vel_xy[0], -self.body_speed_limit_x, self.body_speed_limit_x))
        next_vel_xy = next_vel_xy.at[1].set(jnp.clip(next_vel_xy[1], -self.body_speed_limit_y, self.body_speed_limit_y))

        next_x = jnp.zeros((self.state_dim,), dtype=jnp.float32)
        next_x = next_x.at[:2].set(next_body)
        next_x = next_x.at[2].set(next_psi)
        next_x = next_x.at[3:5].set(next_vel_xy)
        next_x = next_x.at[5].set(next_omega)
        next_x = next_x.at[6].set(next_x_l)
        next_x = next_x.at[7].set(next_x_r)
        next_x = next_x.at[8].set(next_l_l)
        next_x = next_x.at[9].set(next_l_r)
        next_x = next_x.at[10].set(next_mode.astype(jnp.float32))
        next_x = next_x.at[11].set(next_tau)
        return next_x.astype(jnp.float32)

    def jax_model_transition(self, state: Any, action: Any) -> Any:
        return self.jax_transition(state, action)

    def _feet_from_state_np(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        _body, _psi, _vel, x_l, x_r, l_l, l_r, mode, _tau = self._decode_state(state)
        return self._feet_from_components(x_l, x_r, l_l, l_r, mode)

    def get_safety_points(self, state: np.ndarray) -> np.ndarray:
        feet = self._feet_from_state_np(state)
        return np.stack([feet["FL"], feet["FR"], feet["RL"], feet["RR"]], axis=0).astype(np.float32)

    def get_jacobian_safety_points_action(self, state: np.ndarray) -> np.ndarray:
        _body, _psi, _vel, _x_l, _x_r, _l_l, _l_r, mode, _tau = self._decode_state(state)
        J = np.zeros((4, 2, self.act_dim), dtype=np.float32)
        ds_active = 1.0 if mode in (MODE_DS_FL_RR, MODE_DS_FR_RL) else 0.0
        alpha_l = ds_active * float(self.dt) * np.asarray([0.5, 0.5, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        alpha_r = ds_active * float(self.dt) * np.asarray([0.5, -0.5, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        if mode == MODE_DS_FL_RR:
            J[1, 0, :] = alpha_r   # FR
            J[2, 0, :] = -alpha_l  # RL
        elif mode == MODE_DS_FR_RL:
            J[0, 0, :] = alpha_l   # FL
            J[3, 0, :] = -alpha_r  # RR
        return J

    def get_jacobian_safety_points_action_local(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x = _to_np(state, self.state_dim)
        u = np.clip(_to_np(action, self.act_dim), -self.control_limit, self.control_limit)
        base = self.get_safety_points(self.transition(x, u))
        eps = 1e-3
        J = np.zeros((4, 2, self.act_dim), dtype=np.float32)
        for i in range(self.act_dim):
            du = np.zeros((self.act_dim,), dtype=np.float32)
            du[i] = eps
            up = np.clip(u + du, -self.control_limit, self.control_limit)
            um = np.clip(u - du, -self.control_limit, self.control_limit)
            pp = self.get_safety_points(self.transition(x, up))
            pm = self.get_safety_points(self.transition(x, um))
            J[:, :, i] = ((pp - pm) / (2.0 * eps)).astype(np.float32)
        return J

    def jax_safety_points(self, state):
        x = jnp.asarray(state, dtype=jnp.float32).reshape(-1)[: self.state_dim]
        x_l = x[6]
        x_r = x[7]
        l_l = x[8]
        l_r = x[9]
        mode = jnp.mod(jnp.rint(x[10]).astype(jnp.int32), NUM_MODES)
        half_w = jnp.asarray(0.5 * float(self.step_width), dtype=jnp.float32)
        mask = jnp.logical_or(mode == MODE_DS_FL_RR, mode == MODE_QS_AFTER_FL_RR)
        fl = jnp.stack([jnp.where(mask, x_l, x_l + l_l), half_w])
        rl = jnp.stack([jnp.where(mask, x_l - l_l, x_l), half_w])
        rr = jnp.stack([jnp.where(mask, x_r, x_r - l_r), -half_w])
        fr = jnp.stack([jnp.where(mask, x_r + l_r, x_r), -half_w])
        return jnp.stack([fl, fr, rl, rr], axis=0)

    def jax_jacobian_safety_points_action(self, state):
        x = jnp.asarray(state, dtype=jnp.float32).reshape(-1)[: self.state_dim]
        mode = jnp.mod(jnp.rint(x[10]).astype(jnp.int32), NUM_MODES)
        ds_active = jnp.where(jnp.logical_or(mode == MODE_DS_FL_RR, mode == MODE_DS_FR_RL), 1.0, 0.0)
        alpha_l = ds_active * jnp.asarray(float(self.dt), dtype=jnp.float32) * jnp.asarray([0.5, 0.5, 0.0, 1.0, 0.0, 0.0], dtype=jnp.float32)
        alpha_r = ds_active * jnp.asarray(float(self.dt), dtype=jnp.float32) * jnp.asarray([0.5, -0.5, 0.0, 0.0, 1.0, 0.0], dtype=jnp.float32)
        J = jnp.zeros((4, 2, self.act_dim), dtype=jnp.float32)
        J = jnp.where(mode == MODE_DS_FL_RR, J.at[1, 0, :].set(alpha_r).at[2, 0, :].set(-alpha_l), J)
        J = jnp.where(mode == MODE_DS_FR_RL, J.at[0, 0, :].set(alpha_l).at[3, 0, :].set(-alpha_r), J)
        return J

    def jax_jacobian_safety_points_action_local(self, state, action):
        x = jnp.asarray(state, dtype=jnp.float32).reshape(-1)[: self.state_dim]
        u0 = jnp.asarray(action, dtype=jnp.float32).reshape(-1)[: self.act_dim]
        u0 = jnp.clip(u0, -self.control_limit, self.control_limit)

        def next_safety_points(u_in):
            u_clip = jnp.clip(u_in, -self.control_limit, self.control_limit)
            x_next = self.jax_transition(x, u_clip)
            return self.jax_safety_points(x_next)

        return jax.jacfwd(next_safety_points)(u0).transpose(1, 2, 0).astype(jnp.float32)

    def cost(self, state: np.ndarray) -> float:
        body, psi, vel, x_l, x_r, l_l, l_r, mode, _tau = self._decode_state(state)
        feet = self._feet_from_state_np(state)
        goal_vec = self.target - body
        psi_ref = float(np.arctan2(goal_vec[1], goal_vec[0]))
        h_err = _wrap_np(psi - psi_ref)
        support_center = self._mode_support_center_np(feet, mode)
        goal_x_l = float(self.target[0] + self.fore_hind_offset) if mode in (MODE_DS_FL_RR, MODE_QS_AFTER_FL_RR) else float(self.target[0] - self.fore_hind_offset)
        goal_x_r = float(self.target[0] - self.fore_hind_offset) if mode in (MODE_DS_FL_RR, MODE_QS_AFTER_FL_RR) else float(self.target[0] + self.fore_hind_offset)
        goal_feet = self._nominal_feet(self.target, psi_ref)
        foot_err = sum(float(np.sum((feet[leg] - goal_feet[leg]) ** 2)) for leg in ("FL", "FR", "RL", "RR"))
        anchor_err = (x_l - goal_x_l) ** 2 + (x_r - goal_x_r) ** 2
        length_err = (l_l - self.length_nominal) ** 2 + (l_r - self.length_nominal) ** 2
        vel_err = 1.20 * (float(body[1] - self.target[1]) ** 2) + 0.80 * float(vel[1] ** 2) + 0.50 * float(vel[2] ** 2) + 0.25 * max(0.0, -float(vel[0])) ** 2
        return float(
            np.sum(goal_vec ** 2)
            + vel_err
            + 0.08 * (h_err ** 2)
            + 0.80 * anchor_err
            + 1.20 * length_err
            + 0.90 * np.sum((body - support_center) ** 2)
            + 0.10 * foot_err
        )

    def terminal_distance(self, state: np.ndarray) -> float:
        body, psi, _vel, _x_l, _x_r, l_l, l_r, _mode, _tau = self._decode_state(state)
        feet = self._feet_from_state_np(state)
        body_err = float(np.linalg.norm(body - self.target))
        goal_feet = self._nominal_feet(self.target, float(psi))
        foot_err = float(np.mean([np.linalg.norm(feet[leg] - goal_feet[leg]) for leg in ("FL", "FR", "RL", "RR")]))
        length_err = 0.5 * (abs(float(l_l) - self.length_nominal) + abs(float(l_r) - self.length_nominal))
        return body_err + float(self.terminal_foot_penalty) * foot_err + 0.5 * length_err

    def terminal_distance_jax(self, state):
        x = jnp.asarray(state, dtype=jnp.float32).reshape(-1)[: self.state_dim]
        body = x[:2]
        psi = x[2]
        x_l = x[6]
        x_r = x[7]
        l_l = x[8]
        l_r = x[9]
        mode = jnp.mod(jnp.rint(x[10]).astype(jnp.int32), NUM_MODES)
        goal_mid_j = jnp.asarray(self.target, dtype=jnp.float32)
        body_err = jnp.linalg.norm(body - goal_mid_j)
        half_w = 0.5 * float(self.step_width)
        fl = jnp.stack([jnp.where(jnp.logical_or(mode == MODE_DS_FL_RR, mode == MODE_QS_AFTER_FL_RR), x_l, x_l + l_l), jnp.asarray(half_w, dtype=jnp.float32)])
        rl = jnp.stack([jnp.where(jnp.logical_or(mode == MODE_DS_FL_RR, mode == MODE_QS_AFTER_FL_RR), x_l - l_l, x_l), jnp.asarray(half_w, dtype=jnp.float32)])
        rr = jnp.stack([jnp.where(jnp.logical_or(mode == MODE_DS_FL_RR, mode == MODE_QS_AFTER_FL_RR), x_r, x_r - l_r), jnp.asarray(-half_w, dtype=jnp.float32)])
        fr = jnp.stack([jnp.where(jnp.logical_or(mode == MODE_DS_FL_RR, mode == MODE_QS_AFTER_FL_RR), x_r + l_r, x_r), jnp.asarray(-half_w, dtype=jnp.float32)])
        c = jnp.cos(psi)
        s = jnp.sin(psi)
        R = jnp.asarray([[c, -s], [s, c]], dtype=jnp.float32)
        nominal_offsets_j = {leg: jnp.asarray(v, dtype=jnp.float32) for leg, v in self._nominal_offsets().items()}
        goal_feet = {leg: goal_mid_j + R @ nominal_offsets_j[leg] for leg in ("FL", "FR", "RL", "RR")}
        feet = {"FL": fl, "FR": fr, "RL": rl, "RR": rr}
        foot_err = jnp.mean(jnp.asarray([jnp.linalg.norm(feet[leg] - goal_feet[leg]) for leg in ("FL", "FR", "RL", "RR")], dtype=jnp.float32))
        length_err = 0.5 * (jnp.abs(l_l - self.length_nominal) + jnp.abs(l_r - self.length_nominal))
        return body_err + float(self.terminal_foot_penalty) * foot_err + 0.5 * length_err


def make_stepping_stones_energy(env: QuadrupedSteppingStones2DEnv) -> LegacyEnergyFunctional:
    scene = env.scene
    centers = np.asarray(scene.stones_centers, dtype=np.float32)
    radii = np.asarray(scene.stones_radii, dtype=np.float32)
    goal_mid = np.asarray(scene.goal_mid, dtype=np.float32)
    river_x = np.asarray(scene.river_x, dtype=np.float32)
    has_river = bool(getattr(scene, "has_river", True))
    body_speed_limit = float(env.body_speed_limit)
    length_rate_limit = float(env.length_rate_limit)
    nominal_len = float(env.length_nominal)

    def _support_violation_np(p: np.ndarray) -> float:
        p = np.asarray(p, dtype=np.float32).reshape(2)
        d = np.linalg.norm(centers - p[None, :], axis=-1)
        stone_v = max(0.0, float(np.min(np.maximum(d - radii, 0.0)))) if len(centers) > 0 else 1.0
        if has_river:
            if float(p[0]) <= float(river_x[0]) or float(p[0]) >= float(river_x[1]):
                return 0.0
            bank_v = min(abs(float(p[0]) - float(river_x[0])), abs(float(p[0]) - float(river_x[1])))
            return min(stone_v, bank_v)
        return stone_v

    def _support_center_np(feet: Dict[str, np.ndarray], mode: int) -> np.ndarray:
        if mode == MODE_DS_FL_RR:
            return 0.5 * (feet["FL"] + feet["RR"])
        if mode == MODE_DS_FR_RL:
            return 0.5 * (feet["FR"] + feet["RL"])
        return 0.25 * (feet["FL"] + feet["FR"] + feet["RL"] + feet["RR"])

    if jnp is not None:
        centers_j = jnp.asarray(centers, dtype=jnp.float32)
        radii_j = jnp.asarray(radii, dtype=jnp.float32)
        goal_mid_j = jnp.asarray(goal_mid, dtype=jnp.float32)
        river_x_j = jnp.asarray(river_x, dtype=jnp.float32)
        has_river_j = jnp.asarray(1.0 if has_river else 0.0, dtype=jnp.float32)
        half_w_j = jnp.asarray(0.5 * float(env.step_width), dtype=jnp.float32)

        def _decode_j(x):
            x = jnp.asarray(x, dtype=jnp.float32).reshape(-1)
            body = x[:2]
            psi = x[2]
            vel = x[3:6]
            x_l = x[6]
            x_r = x[7]
            l_l = x[8]
            l_r = x[9]
            mode = jnp.mod(jnp.rint(x[10]).astype(jnp.int32), NUM_MODES)
            tau = jnp.clip(x[11], 0.0, 1.0)
            return body, psi, vel, x_l, x_r, l_l, l_r, mode, tau

        def _feet_j(x_l, x_r, l_l, l_r, mode):
            mask = jnp.logical_or(mode == MODE_DS_FL_RR, mode == MODE_QS_AFTER_FL_RR)
            fl = jnp.stack([jnp.where(mask, x_l, x_l + l_l), half_w_j])
            rl = jnp.stack([jnp.where(mask, x_l - l_l, x_l), half_w_j])
            rr = jnp.stack([jnp.where(mask, x_r, x_r - l_r), -half_w_j])
            fr = jnp.stack([jnp.where(mask, x_r + l_r, x_r), -half_w_j])
            return {"FL": fl, "FR": fr, "RL": rl, "RR": rr}

        def _support_violation_j(p):
            d = jnp.linalg.norm(centers_j - p[None, :], axis=-1)
            stone_v = jnp.min(jnp.maximum(d - radii_j, 0.0)) if centers.shape[0] > 0 else jnp.asarray(1.0, dtype=jnp.float32)
            in_bank = jnp.logical_or(p[0] <= river_x_j[0], p[0] >= river_x_j[1])
            bank_v = jnp.minimum(jnp.abs(p[0] - river_x_j[0]), jnp.abs(p[0] - river_x_j[1]))
            river_v = jnp.where(in_bank, 0.0, jnp.minimum(stone_v, bank_v))
            return jnp.where(has_river_j > 0.5, river_v, stone_v)

        def _support_center_j(feet, mode):
            center_flrr = 0.5 * (feet["FL"] + feet["RR"])
            center_frrl = 0.5 * (feet["FR"] + feet["RL"])
            center_qs = 0.25 * (feet["FL"] + feet["FR"] + feet["RL"] + feet["RR"])
            return jnp.where(
                (mode == MODE_DS_FL_RR)[:, None] if mode.ndim > 0 else False,  # unused branch shape hint
                center_flrr,
                center_qs,
            )

        def task_energy(x, u, ctx):
            _ = (u, ctx)
            body, psi, vel, x_l, x_r, l_l, l_r, mode, _tau = _decode_j(x)
            feet = _feet_j(x_l, x_r, l_l, l_r, mode)
            goal_vec = goal_mid_j - body
            psi_ref = jnp.arctan2(goal_vec[1], goal_vec[0])
            h_err = jnp.arctan2(jnp.sin(psi - psi_ref), jnp.cos(psi - psi_ref))
            goal_x_l = jnp.where(
                jnp.logical_or(mode == MODE_DS_FL_RR, mode == MODE_QS_AFTER_FL_RR),
                goal_mid_j[0] + env.fore_hind_offset,
                goal_mid_j[0] - env.fore_hind_offset,
            )
            goal_x_r = jnp.where(
                jnp.logical_or(mode == MODE_DS_FL_RR, mode == MODE_QS_AFTER_FL_RR),
                goal_mid_j[0] - env.fore_hind_offset,
                goal_mid_j[0] + env.fore_hind_offset,
            )
            nominal_feet = {
                "FL": goal_mid_j + jnp.asarray([env.fore_hind_offset, 0.5 * env.step_width], dtype=jnp.float32),
                "FR": goal_mid_j + jnp.asarray([env.fore_hind_offset, -0.5 * env.step_width], dtype=jnp.float32),
                "RL": goal_mid_j + jnp.asarray([-env.fore_hind_offset, 0.5 * env.step_width], dtype=jnp.float32),
                "RR": goal_mid_j + jnp.asarray([-env.fore_hind_offset, -0.5 * env.step_width], dtype=jnp.float32),
            }
            support_center = jnp.where(
                mode == MODE_DS_FL_RR,
                0.5 * (feet["FL"] + feet["RR"]),
                jnp.where(
                    mode == MODE_DS_FR_RL,
                    0.5 * (feet["FR"] + feet["RL"]),
                    0.25 * (feet["FL"] + feet["FR"] + feet["RL"] + feet["RR"]),
                ),
            )
            foot_err = sum(jnp.sum((feet[leg] - nominal_feet[leg]) ** 2) for leg in ("FL", "FR", "RL", "RR"))
            anchor_err = (x_l - goal_x_l) ** 2 + (x_r - goal_x_r) ** 2
            length_err = (l_l - nominal_len) ** 2 + (l_r - nominal_len) ** 2
            vel_err = 1.20 * ((body[1] - goal_mid_j[1]) ** 2) + 0.80 * (vel[1] ** 2) + 0.50 * (vel[2] ** 2) + 0.25 * (jnp.maximum(0.0, -vel[0]) ** 2)
            return (
                jnp.sum((body - goal_mid_j) ** 2)
                + vel_err
                + 0.08 * (h_err ** 2)
                + 0.80 * anchor_err
                + 1.20 * length_err
                + 0.90 * jnp.sum((body - support_center) ** 2)
                + 0.10 * foot_err
            )

        def smooth_energy(x, u, ctx):
            _ = (x, ctx)
            u = jnp.asarray(u, dtype=jnp.float32).reshape(-1)
            return 0.20 * jnp.sum(u[:2] ** 2) + 0.18 * (u[2] ** 2) + 0.20 * jnp.sum(u[3:5] ** 2) + 0.05 * (u[5] ** 2)

        def foothold_energy(x, u, ctx):
            _ = (u, ctx)
            _body, _psi, _vel, x_l, x_r, l_l, l_r, mode, _tau = _decode_j(x)
            feet = _feet_j(x_l, x_r, l_l, l_r, mode)
            return 18.0 * sum(_support_violation_j(feet[leg]) for leg in ("FL", "FR", "RL", "RR"))

        def support_energy(x, u, ctx):
            _ = (u, ctx)
            body, _psi, _vel, x_l, x_r, l_l, l_r, mode, _tau = _decode_j(x)
            feet = _feet_j(x_l, x_r, l_l, l_r, mode)
            center = jnp.where(
                mode == MODE_DS_FL_RR,
                0.5 * (feet["FL"] + feet["RR"]),
                jnp.where(
                    mode == MODE_DS_FR_RL,
                    0.5 * (feet["FR"] + feet["RL"]),
                    0.25 * (feet["FL"] + feet["FR"] + feet["RL"] + feet["RR"]),
                ),
            )
            return 10.0 * jnp.sum((body - center) ** 2)

        def geometry_energy(x, u, ctx):
            _ = (u, ctx)
            body, psi, _vel, x_l, x_r, l_l, l_r, mode, _tau = _decode_j(x)
            feet = _feet_j(x_l, x_r, l_l, l_r, mode)
            c = jnp.cos(psi)
            s = jnp.sin(psi)
            R = jnp.asarray([[c, s], [-s, c]], dtype=jnp.float32)
            nominal = env._nominal_offsets()
            shape = 0.0
            for leg in ("FL", "FR", "RL", "RR"):
                local = R @ (feet[leg] - body)
                shape = shape + jnp.sum((local - jnp.asarray(nominal[leg], dtype=jnp.float32)) ** 2)
            length_reg = (l_l - nominal_len) ** 2 + (l_r - nominal_len) ** 2
            return 1.5 * shape + 12.0 * length_reg

        def step_bound_energy(x, u, ctx):
            _ = (x, ctx)
            u = jnp.asarray(u, dtype=jnp.float32).reshape(-1)
            a_v = jnp.maximum(0.0, jnp.linalg.norm(u[:2]) - env.body_acc_limit)
            dl_v = jnp.maximum(0.0, jnp.linalg.norm(u[3:5]) - length_rate_limit)
            dtau_v = jnp.maximum(0.0, u[5] - env.phase_rate_limit)
            dtau_neg = jnp.maximum(0.0, -u[5])
            yaw_v = jnp.maximum(0.0, jnp.abs(u[2]) - env.yaw_acc_limit)
            return 8.0 * (a_v ** 2 + dl_v ** 2) + 4.0 * (yaw_v ** 2 + dtau_v ** 2 + dtau_neg ** 2)

    else:
        def _decode_np(x):
            x = np.asarray(x, dtype=np.float32).reshape(-1)
            body = x[:2]
            psi = float(x[2])
            vel = x[3:6].astype(np.float32)
            x_l = float(x[6])
            x_r = float(x[7])
            l_l = float(x[8])
            l_r = float(x[9])
            mode = int(np.rint(float(x[10]))) % NUM_MODES
            tau = float(np.clip(x[11], 0.0, 1.0))
            return body, psi, vel, x_l, x_r, l_l, l_r, mode, tau

        def _feet_np(x_l, x_r, l_l, l_r, mode):
            return env._feet_from_components(x_l, x_r, l_l, l_r, mode)

        def task_energy(x, u, ctx):
            _ = (u, ctx)
            body, psi, vel, x_l, x_r, l_l, l_r, mode, _tau = _decode_np(x)
            feet = _feet_np(x_l, x_r, l_l, l_r, mode)
            goal_vec = goal_mid - body
            psi_ref = np.arctan2(float(goal_vec[1]), float(goal_vec[0]))
            h_err = _wrap_np(psi - psi_ref)
            support_center = _support_center_np(feet, mode)
            goal_x_l = float(goal_mid[0] + env.fore_hind_offset) if mode in (MODE_DS_FL_RR, MODE_QS_AFTER_FL_RR) else float(goal_mid[0] - env.fore_hind_offset)
            goal_x_r = float(goal_mid[0] - env.fore_hind_offset) if mode in (MODE_DS_FL_RR, MODE_QS_AFTER_FL_RR) else float(goal_mid[0] + env.fore_hind_offset)
            goal_feet = env._nominal_feet(goal_mid, psi_ref)
            foot_err = sum(float(np.sum((feet[leg] - goal_feet[leg]) ** 2)) for leg in ("FL", "FR", "RL", "RR"))
            anchor_err = (x_l - goal_x_l) ** 2 + (x_r - goal_x_r) ** 2
            length_err = (l_l - nominal_len) ** 2 + (l_r - nominal_len) ** 2
            vel_err = 1.20 * float((body[1] - goal_mid[1]) ** 2) + 0.80 * float(vel[1] ** 2) + 0.50 * float(vel[2] ** 2) + 0.25 * max(0.0, -float(vel[0])) ** 2
            return float(
                np.sum((body - goal_mid) ** 2)
                + vel_err
                + 0.08 * (h_err ** 2)
                + 0.80 * anchor_err
                + 1.20 * length_err
                + 0.90 * np.sum((body - support_center) ** 2)
                + 0.10 * foot_err
            )

        def smooth_energy(x, u, ctx):
            _ = (x, ctx)
            u = np.asarray(u, dtype=np.float32).reshape(-1)
            return 0.20 * float(np.sum(u[:2] ** 2)) + 0.18 * float(u[2] ** 2) + 0.20 * float(np.sum(u[3:5] ** 2)) + 0.05 * float(u[5] ** 2)

        def foothold_energy(x, u, ctx):
            _ = (u, ctx)
            _body, _psi, _vel, x_l, x_r, l_l, l_r, mode, _tau = _decode_np(x)
            feet = _feet_np(x_l, x_r, l_l, l_r, mode)
            return 18.0 * sum(_support_violation_np(feet[leg]) for leg in ("FL", "FR", "RL", "RR"))

        def support_energy(x, u, ctx):
            _ = (u, ctx)
            body, _psi, _vel, x_l, x_r, l_l, l_r, mode, _tau = _decode_np(x)
            feet = _feet_np(x_l, x_r, l_l, l_r, mode)
            center = _support_center_np(feet, mode)
            return 10.0 * float(np.sum((body - center) ** 2))

        def geometry_energy(x, u, ctx):
            _ = (u, ctx)
            body, psi, _vel, x_l, x_r, l_l, l_r, mode, _tau = _decode_np(x)
            feet = _feet_np(x_l, x_r, l_l, l_r, mode)
            R = _rotation_np(-psi)
            nominal = env._nominal_offsets()
            shape = sum(float(np.sum((R @ (feet[leg] - body) - nominal[leg]) ** 2)) for leg in ("FL", "FR", "RL", "RR"))
            length_reg = (l_l - nominal_len) ** 2 + (l_r - nominal_len) ** 2
            return 1.5 * shape + 12.0 * length_reg

        def step_bound_energy(x, u, ctx):
            _ = (x, ctx)
            u = np.asarray(u, dtype=np.float32).reshape(-1)
            a_v = max(0.0, float(np.linalg.norm(u[:2])) - env.body_acc_limit)
            dl_v = max(0.0, float(np.linalg.norm(u[3:5])) - length_rate_limit)
            dtau_v = max(0.0, float(u[5]) - env.phase_rate_limit)
            dtau_neg = max(0.0, -float(u[5]))
            yaw_v = max(0.0, abs(float(u[2])) - env.yaw_acc_limit)
            return 8.0 * (a_v * a_v + dl_v * dl_v) + 4.0 * (yaw_v * yaw_v + dtau_v * dtau_v + dtau_neg * dtau_neg)

    return LegacyEnergyFunctional(
        {
            "task": EnergyTerm(task_energy, 1.0),
            "smooth": EnergyTerm(smooth_energy, 1.0),
            "foothold": EnergyTerm(foothold_energy, 1.0),
            "support": EnergyTerm(support_energy, 1.0),
            "geometry": EnergyTerm(geometry_energy, 1.0),
            "step_bound": EnergyTerm(step_bound_energy, 1.0),
        }
    )

