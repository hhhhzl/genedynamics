from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from genedynamics.tasks.stepping_stones import (
    decode_plan_states,
    estimate_yaw_from_mid,
    pair_to_virtual_feet,
    states_to_lr_pairs,
)


LEG_ORDER = ("FL", "FR", "RL", "RR")
WALK_GROUPS = (("FL",), ("RR",), ("FR",), ("RL",))
TROT_GROUPS = (("FL", "RR"), ("FR", "RL"))

MODE_DS_FL_RR = 0
MODE_QS_AFTER_FL_RR = 1
MODE_DS_FR_RL = 2
MODE_QS_AFTER_FR_RL = 3

LEG_JOINTS = {
    "FL": ("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"),
    "FR": ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"),
    "RL": ("RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"),
    "RR": ("RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"),
}
LEG_TIP_GEOM = {"FL": "FL", "FR": "FR", "RL": "RL", "RR": "RR"}
LEG_TIP_BODY = {"FL": "FL_calf", "FR": "FR_calf", "RL": "RL_calf", "RR": "RR_calf"}

# Sextic swing trajectory basis (same boundary conditions as quadruped_control):
# p(0)=p0, p(1)=pf, p(0.5)=pc, p'(0)=p'(1)=0, p''(0)=p''(1)=0
_SEXTIC_A = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 6.0, 12.0, 20.0, 30.0],
    ],
    dtype=np.float64,
)
_SEXTIC_A_INV = np.linalg.inv(_SEXTIC_A)


def _quat_wxyz_yaw(yaw: float) -> np.ndarray:
    h = 0.5 * float(yaw)
    return np.array([np.cos(h), 0.0, 0.0, np.sin(h)], dtype=np.float64)


def _quat_to_rpy_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    w, x, y, z = q[:4]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.sign(sinp) * (np.pi / 2.0) if abs(sinp) >= 1.0 else np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _wrap_angle(x: float) -> float:
    return float(np.arctan2(np.sin(x), np.cos(x)))


def _interp_angle(a0: float, a1: float, alpha: float) -> float:
    d = _wrap_angle(a1 - a0)
    return float(a0 + float(alpha) * d)


def _smoothstep(alpha: float) -> float:
    a = float(np.clip(alpha, 0.0, 1.0))
    return a * a * (3.0 - 2.0 * a)


def _swing_legs_from_mode(mode: int) -> Tuple[str, ...]:
    m = int(mode) % 4
    if m in (MODE_DS_FL_RR, MODE_QS_AFTER_FL_RR):
        return ("FR", "RL")
    return ("FL", "RR")


@dataclass
class MinimalFollowerConfig:
    gait: str = "walk"  # walk | trot
    sim_dt: float = 0.01
    phase_steps: int = 36
    settle_steps: int = 30
    goal_hold_steps: int = 40  # governed path only: hold final pose so the base-PD body settles to goal

    # Geometry / nominal posture
    step_width: float = 0.28
    centerline_y: float = 0.0  # corridor stepping: world y of s–ey midline (matches env.centerline_y)
    leg_half_length: float = 0.18
    x_f_nominal: float = 0.18
    x_r_nominal: float = -0.18
    y_L_nominal: float = 0.15
    y_R_nominal: float = -0.15
    base_height_offset: float = 0.01
    base_z_min: float = 0.23
    min_foot_z: float = 0.015
    swing_height: float = 0.045
    # Match quadruped_control timing: keep stance for early phase, swing late.
    swing_stance_phase: float = 0.5
    swing_use_stance_gate: bool = True
    min_swing_clearance: float = 0.0
    min_swing_clearance_first_n_intervals: int = 0

    # Base stabilization
    use_base_pd: bool = True
    base_kp_xy: float = 44.0
    base_kd_xy: float = 11.0
    base_kp_z: float = 260.0
    base_kd_z: float = 34.0
    base_kp_rp: float = 95.0
    base_kd_rp: float = 12.0
    base_kp_yaw: float = 35.0
    base_kd_yaw: float = 6.0
    base_weight_comp: float = 1.0
    base_force_xy_clip: float = 90.0
    base_force_z_clip: float = 260.0
    base_torque_clip: float = 48.0
    base_support_contact_min: int = 2
    base_support_scale_min: float = 0.0
    use_dynamic_base_z_ref: bool = True
    base_target_clearance_from_feet: float = 0.31
    use_stance_force_distribution: bool = True
    use_stance_qp: bool = True
    stance_qp_solver_order: Tuple[str, ...] = ("clarabel", "osqp", "cvxopt")
    stance_fd_lambda: float = 1e-3
    stance_fd_mu: float = 0.6
    stance_fd_fz_min: float = 5.0
    stance_fd_fz_max: float = 220.0
    stance_fd_torque_blend: float = 0.65

    # Foot task-space control
    swing_kp_xy: float = 170.0
    swing_kp_z: float = 230.0
    swing_kd_xy: float = 14.0
    swing_kd_z: float = 18.0
    use_swing_jointspace_tracking: bool = True
    swing_ik_kp_pos: float = 18.0
    swing_ik_damping: float = 0.010
    swing_joint_kp: float = 80.0
    swing_joint_kd: float = 5.0
    swing_q_step_clip: float = 0.22
    swing_qd_ref_clip: float = 8.0
    swing_jointspace_tau_blend: float = 0.35
    stance_kp_xy: float = 110.0
    stance_kp_z: float = 135.0
    stance_kd_xy: float = 24.0
    stance_kd_z: float = 16.0
    flat_only_foot_lock: bool = False
    flat_lock_stance_kp_xy_scale: float = 1.8
    flat_lock_stance_kp_z_scale: float = 1.25
    flat_lock_stance_kd_xy_scale: float = 1.35
    flat_lock_stance_kd_z_scale: float = 1.20
    flat_lock_max_anchor_error_xy: float = 0.06

    # Joint regularization only
    joint_kp: float = 26.0
    joint_kd: float = 1.4
    joint_torque_clip: float = 3.0
    task_torque_clip: float = 12.0

    # Touchdown / release logic
    touchdown_force_thresh: float = 12.0
    touchdown_force_thresh_stone: float = 12.0
    touchdown_force_thresh_bank: float = 10.0
    touchdown_force_thresh_river: float = 8.0
    touchdown_tangent_speed_thresh: float = 0.20
    touchdown_xy_tol: float = 0.06
    touchdown_stable_steps: int = 2
    post_touchdown_hold_steps: int = 8
    strong_touchdown_force_thresh: float = 28.0
    strong_touchdown_force_thresh_stone: float = 28.0
    strong_touchdown_force_thresh_bank: float = 22.0
    strong_touchdown_force_thresh_river: float = 18.0
    touchdown_counter_decay_on_contact: int = 1
    release_force_thresh: float = 5.0
    release_height_margin: float = 0.015
    release_stable_steps: int = 3
    allow_early_switch: bool = False
    allow_early_switch_on_contact: bool = False
    early_switch_contact_alpha_min: float = 0.55

    # Safety / saturation
    roll_pitch_abort_rad: float = 0.45
    max_anchor_error_xy: float = 0.04
    max_swing_tracking_error_xy: float = 0.08
    swing_xy_finish_alpha: float = 0.80
    swing_z_only_tail: float = 0.20
    stance_press_z_offset: float = -0.004
    stance_plan_blend: float = 0.65
    uniform_base_speed: bool = True
    base_speed_mps: float = 0.16
    support_contact_abort_steps: int = 80
    touchdown_timeout_alpha: float = 0.90
    touchdown_timeout_steps: int = 16
    force_interval_end_on_touchdown_timeout: bool = True
    abort_after_consecutive_timeouts: int = 2


@dataclass
class PhaseState:
    swing_legs: Tuple[str, ...]
    stance_legs: Tuple[str, ...]
    start_mid_xy: np.ndarray
    goal_mid_xy: np.ndarray
    start_yaw: float
    goal_yaw: float
    swing_start: Dict[str, np.ndarray]
    swing_goal: Dict[str, np.ndarray]
    swing_coeff: Dict[str, np.ndarray]
    stance_anchor: Dict[str, np.ndarray]


class SteppingWalkFollowerMinimal:
    """
    Minimal follower.

    Compared with the first minimal version:
    - uses mj_contactForce for per-leg normal/tangential contact estimation,
    - uses more robust touchdown / release gating,
    - keeps a single control chain only:
        gait -> foot target -> task-space torques (+ light joint regularization).

    It intentionally does NOT include the original contact QP / centroidal wrench stack,
    because the goal is to keep one dominant controller while you debug locomotion.
    """

    def __init__(
        self,
        model_xml_path: Optional[str] = None,
        cfg: Optional[MinimalFollowerConfig] = None,
        stepping_scene: Optional[Dict[str, Any]] = None,
        gait: Optional[str] = None,
        sim_dt: Optional[float] = None,
        phase_steps: Optional[int] = None,
    ) -> None:
        import mujoco
        from genedynamics.envs.utils.mujoco_model_generator import create_go2_sim_xml_with_stepping_scene
        from genedynamics.robots.registry import _get_go2_path

        self.mujoco = mujoco
        self.cfg = cfg or MinimalFollowerConfig()
        if gait is not None:
            self.cfg.gait = str(gait)
        if sim_dt is not None:
            self.cfg.sim_dt = float(sim_dt)
        if phase_steps is not None:
            self.cfg.phase_steps = int(phase_steps)

        self.stepping_scene = stepping_scene or None

        if model_xml_path is None:
            p = Path(_get_go2_path() or "")
            if not p.exists():
                raise FileNotFoundError("Go2 model not found. Set MUJOCO_MENAGERIE_PATH.")
            scene = p.parent / "scene.xml"
            model_xml_path = str(scene if scene.exists() else p)

        temp_xml_path: Optional[Path] = None
        if self.stepping_scene:
            src = Path(str(model_xml_path))
            temp_xml_path = src.parent / "_walk_follow_scene_temp_minimal_v2.xml"
            create_go2_sim_xml_with_stepping_scene(
                str(temp_xml_path),
                stepping_scene=self.stepping_scene,
                go2_xml_path=str(src),
            )
            model_xml_path = str(temp_xml_path)

        try:
            self.model = mujoco.MjModel.from_xml_path(str(model_xml_path))
        finally:
            if temp_xml_path is not None:
                try:
                    os.unlink(temp_xml_path)
                except OSError:
                    pass

        self.model.opt.timestep = float(self.cfg.sim_dt)
        self.data = mujoco.MjData(self.model)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self._build_indices()
        self._home_qpos = np.array(self.data.qpos, copy=True)
        self._base_height = float(self._home_qpos[2])
        self._total_mass = float(np.sum(self.model.body_mass[1:]))

        self._scene_bank_top_z = 0.0
        self._scene_river_depth = 0.10
        self._scene_stone_top_z = 0.035
        centers = np.asarray((self.stepping_scene or {}).get("stones_centers", []), dtype=np.float64).reshape(-1, 2)
        has_river = bool((self.stepping_scene or {}).get("has_river", True))
        self._flat_ground_scene = (centers.shape[0] == 0) and (not has_river)

        self._contact_wrench_by_leg: Dict[str, np.ndarray] = {leg: np.zeros(6, dtype=np.float64) for leg in LEG_ORDER}
        self._normal_force_by_leg: Dict[str, float] = {leg: 0.0 for leg in LEG_ORDER}
        self._tangent_speed_by_leg: Dict[str, float] = {leg: 0.0 for leg in LEG_ORDER}
        self._ground_contact_set: set[str] = set()
        self._contact_surface_by_leg: Dict[str, str] = {leg: "none" for leg in LEG_ORDER}
        self._contact_surface_force_by_leg: Dict[str, float] = {leg: 0.0 for leg in LEG_ORDER}

    # ------------------------------------------------------------------
    # Compatibility wrappers
    # ------------------------------------------------------------------
    @property
    def sim_dt(self) -> float:
        return float(self.cfg.sim_dt)

    @property
    def phase_steps(self) -> int:
        return int(self.cfg.phase_steps)

    def run(
        self,
        plan_states: np.ndarray,
        *,
        max_segments: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.follow_plan(plan_states, max_segments=max_segments)

    def rollout(
        self,
        plan_states: np.ndarray,
        *,
        max_segments: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.follow_plan(plan_states, max_segments=max_segments)

    def reset(self) -> None:
        mujoco = self.mujoco
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._clear_contact_cache()

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def _mode_intervals(self, modes: np.ndarray, n_seg: int) -> List[Tuple[int, int, int]]:
        if n_seg <= 0:
            return []
        mm = np.asarray(modes, dtype=np.int32).reshape(-1)
        if mm.size < n_seg:
            mm = np.pad(mm, (0, n_seg - mm.size), mode="edge")
        out: List[Tuple[int, int, int]] = []
        start = 0
        curr = int(mm[0])
        for k in range(1, n_seg):
            mk = int(mm[k])
            if mk != curr:
                out.append((start, k - 1, curr))
                start = k
                curr = mk
        out.append((start, n_seg - 1, curr))
        return out

    def _build_path_arclen(self, mids_xy: np.ndarray) -> np.ndarray:
        mids_xy = np.asarray(mids_xy, dtype=np.float64)
        if mids_xy.shape[0] <= 1:
            return np.zeros((mids_xy.shape[0],), dtype=np.float64)
        d = np.linalg.norm(mids_xy[1:] - mids_xy[:-1], axis=1)
        return np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(d, dtype=np.float64)], axis=0)

    def _interp_path_by_s(
        self,
        mids_xy: np.ndarray,
        yaws: np.ndarray,
        s_cum: np.ndarray,
        s_query: float,
    ) -> Tuple[np.ndarray, float]:
        mids_xy = np.asarray(mids_xy, dtype=np.float64)
        yaws = np.asarray(yaws, dtype=np.float64).reshape(-1)
        s_cum = np.asarray(s_cum, dtype=np.float64).reshape(-1)
        if mids_xy.shape[0] <= 1:
            return mids_xy[0].copy(), float(yaws[0])
        sq = float(np.clip(s_query, 0.0, float(s_cum[-1])))
        idx = int(np.searchsorted(s_cum, sq, side="right") - 1)
        idx = max(0, min(idx, mids_xy.shape[0] - 2))
        seg_len = float(s_cum[idx + 1] - s_cum[idx])
        if seg_len <= 1e-9:
            frac = 0.0
        else:
            frac = float((sq - float(s_cum[idx])) / seg_len)
        xy = (1.0 - frac) * mids_xy[idx] + frac * mids_xy[idx + 1]
        yaw = _interp_angle(float(yaws[idx]), float(yaws[idx + 1]), frac)
        return np.asarray(xy, dtype=np.float64), float(yaw)

    def _swing_time_from_phase(self, phase: float) -> float:
        """Map gait phase to swing trajectory time, like quadruped_control."""
        p = float(np.clip(phase, 0.0, 1.0))
        if not bool(self.cfg.swing_use_stance_gate):
            return p
        stance_phase = float(np.clip(self.cfg.swing_stance_phase, 0.0, 0.95))
        slope = 1.0 / max(1.0 - stance_phase, 1e-6)
        y_intercept = 1.0 - slope
        t = slope * p + y_intercept
        return float(np.clip(t, 0.0, 1.0))

    def _swing_clearance_for_interval(self, interval_idx: int) -> float:
        c = float(max(0.0, self.cfg.min_swing_clearance))
        if c <= 0.0:
            return 0.0
        n = int(self.cfg.min_swing_clearance_first_n_intervals)
        if n <= 0:
            return c
        return c if int(interval_idx) < n else 0.0

    def _touchdown_force_threshold(self, surface: str, *, strong: bool) -> float:
        s = str(surface or "none")
        if strong:
            if s == "stone":
                return float(max(0.0, self.cfg.strong_touchdown_force_thresh_stone))
            if s == "bank":
                return float(max(0.0, self.cfg.strong_touchdown_force_thresh_bank))
            if s == "river":
                return float(max(0.0, self.cfg.strong_touchdown_force_thresh_river))
            return float(max(0.0, self.cfg.strong_touchdown_force_thresh))
        if s == "stone":
            return float(max(0.0, self.cfg.touchdown_force_thresh_stone))
        if s == "bank":
            return float(max(0.0, self.cfg.touchdown_force_thresh_bank))
        if s == "river":
            return float(max(0.0, self.cfg.touchdown_force_thresh_river))
        return float(max(0.0, self.cfg.touchdown_force_thresh))

    def follow_plan(
        self,
        plan_states: np.ndarray,
        *,
        max_segments: Optional[int] = None,
    ) -> Dict[str, Any]:
        mids_xy, yaws, leg_goals, modes, taus = self._decode_plan(plan_states)
        n_seg = max(0, len(mids_xy) - 1)
        if max_segments is not None:
            n_seg = min(n_seg, int(max_segments))

        if n_seg <= 0:
            return self._empty_rollout()

        qpos_hist: List[np.ndarray] = []
        qvel_hist: List[np.ndarray] = []
        ctrl_hist: List[np.ndarray] = []
        rpy_hist: List[np.ndarray] = []
        contact_hist: List[List[str]] = []
        debug_hist: List[Dict[str, Any]] = []
        swing_ref_hist: Dict[str, List[np.ndarray]] = {leg: [] for leg in LEG_ORDER}

        self._set_base_pose(mids_xy[0], yaws[0])
        self._warm_start_stance(steps=self.cfg.settle_steps)

        terminated = False
        term_reason = None
        timeout_streak = 0
        low_support_counter = 0
        interval_stats: List[Dict[str, Any]] = []
        executed_footholds: List[Dict[str, Any]] = []

        intervals = self._mode_intervals(modes, n_seg)
        s_cum = self._build_path_arclen(mids_xy)
        total_len = float(s_cum[-1]) if s_cum.size > 0 else 0.0
        dt = float(self.cfg.sim_dt)
        total_substeps = max(1, n_seg * max(1, int(self.cfg.phase_steps)))
        if bool(self.cfg.uniform_base_speed) and total_len > 1e-6:
            speed_nom = float(self.cfg.base_speed_mps)
            if speed_nom <= 1e-6:
                speed_nom = total_len / max(float(total_substeps) * dt, 1e-6)
            ds = max(0.0, speed_nom * dt)
        else:
            ds = 0.0
        s_prog = 0.0

        for interval_idx, (seg_start, seg_end, mode_k) in enumerate(intervals):
            swing_legs = _swing_legs_from_mode(mode_k)
            phase = self._make_phase_state(
                swing_legs=swing_legs,
                mid0=mids_xy[seg_start],
                mid1=mids_xy[seg_end + 1],
                yaw0=yaws[seg_start],
                yaw1=yaws[seg_end + 1],
                leg_goals0=leg_goals[seg_start],
                leg_goals1=leg_goals[seg_end + 1],
                clearance_extra=self._swing_clearance_for_interval(interval_idx),
            )

            tau_start = float(taus[seg_start]) if seg_start < len(taus) else 0.0
            if (seg_end + 1) < len(modes) and int(modes[seg_end + 1]) != int(mode_k):
                tau_end = 1.0
            else:
                tau_end = float(np.max(taus[seg_start : min(seg_end + 2, len(taus))])) if len(taus) > 0 else 1.0
            tau_end = max(tau_end, tau_start + 1e-3)
            # If planner stores tau=1.0 at all boundaries, recover a local phase [0,1].
            if tau_start >= 1.0 - 1e-6 or (tau_end - tau_start) <= 1e-4:
                tau_start, tau_end = 0.0, 1.0

            touchdown_counter = {leg: 0 for leg in phase.swing_legs}
            touchdown_latched = {leg: False for leg in phase.swing_legs}
            hold_counter = 0
            timeout_counter = 0
            interval_timed_out = False
            touchdown_any = False
            swing_ground_contact_any = False
            interval_edges = max(1, seg_end - seg_start + 1)
            interval_steps = interval_edges * max(1, int(self.cfg.phase_steps))
            phase_duration = max(float(self.cfg.sim_dt), float(interval_steps) * dt)
            for sub_global in range(interval_steps):
                alpha_global = float(sub_global + 1) / float(max(1, interval_steps))
                phase_tau = float(np.clip((1.0 - alpha_global) * tau_start + alpha_global * tau_end, 0.0, 1.0))
                alpha_local = float(np.clip((phase_tau - tau_start) / max(tau_end - tau_start, 1e-6), 0.0, 1.0))
                alpha_s = self._swing_time_from_phase(alpha_local)
                swing_duration = phase_duration
                if bool(self.cfg.swing_use_stance_gate):
                    stance_phase = float(np.clip(self.cfg.swing_stance_phase, 0.0, 0.95))
                    swing_duration = phase_duration * max(1.0 - stance_phase, 1e-6)

                if ds > 0.0:
                    s_prog = min(total_len, s_prog + ds)
                    base_xy_ref, base_yaw_ref = self._interp_path_by_s(mids_xy, yaws, s_cum, s_prog)
                else:
                    edge_s = _smoothstep(alpha_global)
                    base_xy_ref = (1.0 - edge_s) * phase.start_mid_xy + edge_s * phase.goal_mid_xy
                    base_yaw_ref = _interp_angle(phase.start_yaw, phase.goal_yaw, edge_s)

                target_feet = self._build_foot_targets(
                    phase,
                    alpha_s,
                    phase_duration=swing_duration,
                )
                foot_vel_ref = self._build_foot_velocity_refs(
                    phase,
                    alpha_s,
                    phase_duration=swing_duration,
                )

                support_contacts = len(self._ground_contact_legs())
                base_wrench: Optional[Tuple[np.ndarray, np.ndarray]] = None
                if self.cfg.use_base_pd:
                    base_wrench = self._apply_base_pd(base_xy_ref, base_yaw_ref, support_contacts)
                else:
                    self.data.qfrc_applied[:] = 0.0

                ctrl = self._compose_leg_torques(
                    phase=phase,
                    target_feet=target_feet,
                    foot_vel_ref=foot_vel_ref,
                    base_wrench=base_wrench,
                )
                self.data.ctrl[:] = ctrl
                self.mujoco.mj_step(self.model, self.data)
                self._update_contact_measurements()
                ctrl_hist.append(ctrl.copy())
                for leg in LEG_ORDER:
                    if leg in phase.swing_legs:
                        swing_ref_hist[leg].append(np.asarray(target_feet[leg], dtype=np.float64).copy())
                    else:
                        swing_ref_hist[leg].append(np.array([np.nan, np.nan, np.nan], dtype=np.float64))

                for leg in phase.swing_legs:
                    if touchdown_latched[leg]:
                        continue
                    if self._strong_touchdown_ready(leg):
                        touchdown_latched[leg] = True
                        touchdown_counter[leg] = int(self.cfg.touchdown_stable_steps)
                    elif self._touchdown_candidate_ready(leg, target_feet[leg]):
                        touchdown_counter[leg] += 1
                    else:
                        if leg in self._ground_contact_legs():
                            touchdown_counter[leg] = max(
                                0,
                                touchdown_counter[leg] - int(self.cfg.touchdown_counter_decay_on_contact),
                            )
                        else:
                            touchdown_counter[leg] = 0
                    if touchdown_counter[leg] >= int(self.cfg.touchdown_stable_steps):
                        touchdown_latched[leg] = True

                touchdown_all = all(
                    touchdown_latched[leg] or touchdown_counter[leg] >= int(self.cfg.touchdown_stable_steps)
                    for leg in phase.swing_legs
                ) if phase.swing_legs else True
                touchdown_any = touchdown_any or any(bool(v) for v in touchdown_latched.values())

                rpy = _quat_to_rpy_wxyz(self.data.qpos[3:7])
                contacts = self._ground_contact_legs()
                swing_ground_contact_any = swing_ground_contact_any or any(leg in contacts for leg in phase.swing_legs)
                swing_ground_contact_all = all(leg in contacts for leg in phase.swing_legs) if phase.swing_legs else True
                hold_ready = bool(touchdown_all)
                if bool(self.cfg.allow_early_switch_on_contact):
                    hold_ready = hold_ready or (
                        bool(swing_ground_contact_all)
                        and alpha_local >= float(np.clip(self.cfg.early_switch_contact_alpha_min, 0.0, 1.0))
                    )
                if hold_ready:
                    hold_counter += 1
                else:
                    hold_counter = 0
                qpos_hist.append(self.data.qpos.copy())
                qvel_hist.append(self.data.qvel.copy())
                rpy_hist.append(rpy.copy())
                contact_hist.append(contacts)
                pseudo_seg = seg_start + min(interval_edges - 1, int((sub_global * interval_edges) / max(1, interval_steps)))
                pseudo_sub = int(sub_global % max(1, int(self.cfg.phase_steps)))
                debug_hist.append(
                    {
                        "segment": pseudo_seg,
                        "interval": interval_idx,
                        "substep": pseudo_sub,
                        "phase_alpha": alpha_global,
                        "phase_tau": phase_tau,
                        "mode": mode_k,
                        "swing_legs": phase.swing_legs,
                        "stance_legs": phase.stance_legs,
                        "base_xy_ref": base_xy_ref.copy(),
                        "base_yaw_ref": base_yaw_ref,
                        "base_rpy": rpy.copy(),
                        "normal_force": {leg: float(self._normal_force_by_leg[leg]) for leg in LEG_ORDER},
                        "tangent_speed": {leg: float(self._tangent_speed_by_leg[leg]) for leg in LEG_ORDER},
                        "contact_surface": {leg: str(self._contact_surface_by_leg[leg]) for leg in LEG_ORDER},
                        "touchdown_counter": touchdown_counter.copy(),
                        "touchdown_latched": touchdown_latched.copy(),
                        "touchdown_timeout_counter": int(timeout_counter),
                    }
                )

                if support_contacts < int(self.cfg.base_support_contact_min):
                    low_support_counter += 1
                else:
                    low_support_counter = 0
                if low_support_counter >= int(max(1, self.cfg.support_contact_abort_steps)):
                    terminated = True
                    term_reason = "support_contact_abort"
                    break

                if np.max(np.abs(rpy[:2])) > float(self.cfg.roll_pitch_abort_rad):
                    terminated = True
                    term_reason = "roll_pitch_abort"
                    break

                if not touchdown_all:
                    if alpha_local >= float(np.clip(self.cfg.touchdown_timeout_alpha, 0.0, 1.0)):
                        timeout_counter += 1
                    else:
                        timeout_counter = 0
                else:
                    timeout_counter = 0
                if timeout_counter >= int(max(1, self.cfg.touchdown_timeout_steps)):
                    interval_timed_out = True
                    if bool(self.cfg.force_interval_end_on_touchdown_timeout):
                        break
                    terminated = True
                    term_reason = "touchdown_timeout_abort"
                    break

                early_switch_ready = bool(touchdown_all)
                if bool(self.cfg.allow_early_switch_on_contact):
                    early_switch_ready = early_switch_ready or (
                        bool(swing_ground_contact_all)
                        and alpha_local >= float(np.clip(self.cfg.early_switch_contact_alpha_min, 0.0, 1.0))
                    )
                if bool(self.cfg.allow_early_switch) and early_switch_ready and hold_counter >= int(self.cfg.post_touchdown_hold_steps):
                    break

            interval_stats.append(
                {
                    "interval": int(interval_idx),
                    "mode": int(mode_k),
                    "touchdown_any": bool(touchdown_any),
                    "swing_contact_any": bool(swing_ground_contact_any),
                    "timed_out": bool(interval_timed_out),
                }
            )
            # Symmetric executed-foothold logging (purely additive; no control change) so
            # raw and governed rollouts are scored on the SAME executed-foothold criterion.
            for leg in phase.swing_legs:
                landed = self._current_foot_pos(leg)
                executed_footholds.append(
                    {
                        "step": int(interval_idx),
                        "leg": str(leg),
                        "landed_xy": [float(landed[0]), float(landed[1])],
                        "landed_z": float(landed[2]),
                    }
                )
            if interval_timed_out:
                timeout_streak += 1
            else:
                timeout_streak = 0
            if timeout_streak >= int(max(1, self.cfg.abort_after_consecutive_timeouts)):
                terminated = True
                term_reason = "consecutive_touchdown_timeouts"

            if terminated:
                break

        goal_xy = np.asarray(mids_xy[min(n_seg, len(mids_xy) - 1)], dtype=np.float64)
        if len(qpos_hist) > 0:
            base_xy_end = np.asarray(qpos_hist[-1][:2], dtype=np.float64)
            base_xy_start = np.asarray(qpos_hist[0][:2], dtype=np.float64)
            goal_error_xy = float(np.linalg.norm(base_xy_end - goal_xy))
            progress_xy = float(np.linalg.norm(base_xy_end - base_xy_start))
        else:
            goal_error_xy = float("nan")
            progress_xy = 0.0
        rpy_arr = np.asarray(rpy_hist, dtype=np.float64) if len(rpy_hist) > 0 else np.zeros((0, 3), dtype=np.float64)
        interval_timeout_ratio = (
            float(np.mean([1.0 if bool(x["timed_out"]) else 0.0 for x in interval_stats]))
            if len(interval_stats) > 0
            else 0.0
        )
        summary = {
            "goal_error_xy": goal_error_xy,
            "base_progress_xy": progress_xy,
            "pitch_abs_max": float(np.max(np.abs(rpy_arr[:, 1]))) if rpy_arr.size > 0 else 0.0,
            "roll_abs_max": float(np.max(np.abs(rpy_arr[:, 0]))) if rpy_arr.size > 0 else 0.0,
            "interval_count": int(len(interval_stats)),
            "interval_timeout_ratio": float(interval_timeout_ratio),
            "touchdown_any_ratio": float(np.mean([1.0 if bool(x["touchdown_any"]) else 0.0 for x in interval_stats]))
            if len(interval_stats) > 0
            else 0.0,
            "swing_contact_any_ratio": float(np.mean([1.0 if bool(x["swing_contact_any"]) else 0.0 for x in interval_stats]))
            if len(interval_stats) > 0
            else 0.0,
        }

        return {
            "qpos": np.asarray(qpos_hist),
            "qvel": np.asarray(qvel_hist),
            "ctrl": np.asarray(ctrl_hist),
            "base_rpy": np.asarray(rpy_hist),
            "contact_legs": contact_hist,
            "debug": debug_hist,
            "swing_ref": {leg: np.asarray(vals, dtype=np.float64) for leg, vals in swing_ref_hist.items()},
            "interval_stats": interval_stats,
            "executed_footholds": executed_footholds,
            "summary": summary,
            "terminated": terminated,
            "termination_reason": term_reason,
        }

    # ------------------------------------------------------------------
    # Governed gait reference (Rec. 2/3): clean hook for the stepping
    # reference governor.  Each StepPhase is ONE physical step (walk: one
    # swing leg; the other three are frozen stance anchors).  Reuses the
    # same low-level control + touchdown/timeout logic as ``follow_plan``;
    # ``follow_plan`` itself is left untouched.
    # ------------------------------------------------------------------
    def _make_phase_state_governed(
        self,
        *,
        swing_legs: Tuple[str, ...],
        stance_legs: Tuple[str, ...],
        mid0: np.ndarray,
        mid1: np.ndarray,
        yaw0: float,
        yaw1: float,
        swing_goal_world: Dict[str, np.ndarray],
        stance_anchor_world: Dict[str, np.ndarray],
        clearance_extra: float = 0.0,
    ) -> PhaseState:
        swing_start = {leg: self._current_foot_pos(leg) for leg in swing_legs}
        swing_goal: Dict[str, np.ndarray] = {}
        swing_coeff: Dict[str, np.ndarray] = {}
        for leg in swing_legs:
            g = np.asarray(swing_goal_world[leg], dtype=np.float64).reshape(-1).copy()
            if g.shape[0] < 3:
                g = np.array([g[0], g[1], self._terrain_height_at(g[:2])], dtype=np.float64)
            # Re-project onto the safe stone interior for robustness, then set terrain Z.
            if self.stepping_scene:
                g[:2] = self._project_to_safe_support(g)[:2]
            g[2] = max(float(self.cfg.min_foot_z), float(self._terrain_height_at(g[:2])))
            apex_z = max(
                float(self.cfg.min_foot_z),
                float(max(swing_start[leg][2], g[2]) + max(self.cfg.swing_height, float(clearance_extra))),
            )
            swing_coeff[leg] = self._swing_poly_coeff(swing_start[leg], g, apex_z=float(apex_z))
            swing_goal[leg] = g
        stance_anchor: Dict[str, np.ndarray] = {}
        for leg in stance_legs:
            a = np.asarray(stance_anchor_world[leg], dtype=np.float64).reshape(-1).copy()
            if a.shape[0] < 3:
                a = np.array([a[0], a[1], 0.0], dtype=np.float64)
            stance_anchor[leg] = self._stance_anchor_target(a)
        return PhaseState(
            swing_legs=tuple(swing_legs),
            stance_legs=tuple(stance_legs),
            start_mid_xy=np.asarray(mid0, dtype=np.float64).reshape(2).copy(),
            goal_mid_xy=np.asarray(mid1, dtype=np.float64).reshape(2).copy(),
            start_yaw=float(yaw0),
            goal_yaw=float(yaw1),
            swing_start=swing_start,
            swing_goal=swing_goal,
            swing_coeff=swing_coeff,
            stance_anchor=stance_anchor,
        )

    def follow_gait_reference(
        self,
        gait_ref: Any,
        *,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute a governed :class:`GaitReference` (duck-typed: ``.phases`` with
        ``swing_legs / stance_legs / body_mid_start / body_mid_goal / yaw_start /
        yaw_goal / foot_goal``).

        Returns the same result structure as :meth:`follow_plan`, plus a ``governed``
        block (per-step stats, executed footholds, governor diagnostics) used by the
        config-driven exec-SSR runner.
        """
        phases = list(getattr(gait_ref, "phases", []) or [])
        if max_steps is not None:
            phases = phases[: int(max_steps)]
        if len(phases) == 0:
            out = self._empty_rollout()
            out["governed"] = {"used": True, "n_steps": 0, "step_stats": [], "executed_footholds": []}
            return out

        qpos_hist: List[np.ndarray] = []
        qvel_hist: List[np.ndarray] = []
        ctrl_hist: List[np.ndarray] = []
        rpy_hist: List[np.ndarray] = []
        contact_hist: List[List[str]] = []
        debug_hist: List[Dict[str, Any]] = []
        swing_ref_hist: Dict[str, List[np.ndarray]] = {leg: [] for leg in LEG_ORDER}

        p0 = phases[0]
        self._set_base_pose(np.asarray(p0.body_mid_start, dtype=np.float64).reshape(2), float(p0.yaw_start))

        # Posture-holding warm-up: stand cleanly on all four feet (base PD + stance control)
        # rather than a zero-torque settle, which lets the legs splay before the gait engages.
        hold_phase = self._make_phase_state_governed(
            swing_legs=(), stance_legs=LEG_ORDER,
            mid0=p0.body_mid_start, mid1=p0.body_mid_start,
            yaw0=p0.yaw_start, yaw1=p0.yaw_start,
            swing_goal_world={},
            stance_anchor_world={leg: self._current_foot_pos(leg) for leg in LEG_ORDER},
        )
        hold_dur = max(float(self.cfg.sim_dt), float(self.cfg.settle_steps) * float(self.cfg.sim_dt))
        for _ in range(int(max(0, self.cfg.settle_steps))):
            support_contacts = len(self._ground_contact_legs())
            if self.cfg.use_base_pd:
                base_wrench = self._apply_base_pd(p0.body_mid_start, float(p0.yaw_start), support_contacts)
            else:
                self.data.qfrc_applied[:] = 0.0
                base_wrench = None
            tf = self._build_foot_targets(hold_phase, 1.0, phase_duration=hold_dur)
            fvr = self._build_foot_velocity_refs(hold_phase, 1.0, phase_duration=hold_dur)
            ctrl = self._compose_leg_torques(phase=hold_phase, target_feet=tf, foot_vel_ref=fvr, base_wrench=base_wrench)
            self.data.ctrl[:] = ctrl
            self.mujoco.mj_step(self.model, self.data)
            self._update_contact_measurements()

        # Stance anchors start at the robot's *actual* feet (no startup yank), then get
        # frozen to the governed stone targets once a leg has stepped onto its stone.
        executed_anchor: Dict[str, np.ndarray] = {leg: self._current_foot_pos(leg) for leg in LEG_ORDER}

        dt = float(self.cfg.sim_dt)
        terminated = False
        term_reason: Optional[str] = None
        timeout_streak = 0
        low_support_counter = 0
        step_stats: List[Dict[str, Any]] = []
        executed_footholds: List[Dict[str, Any]] = []

        for step_idx, ph in enumerate(phases):
            swing_legs = tuple(ph.swing_legs)
            stance_legs = tuple(ph.stance_legs)
            swing_goal_world = {leg: np.asarray(ph.foot_goal[leg], dtype=np.float64) for leg in swing_legs}
            stance_anchor_world = {leg: executed_anchor[leg] for leg in stance_legs}
            phase = self._make_phase_state_governed(
                swing_legs=swing_legs,
                stance_legs=stance_legs,
                mid0=ph.body_mid_start,
                mid1=ph.body_mid_goal,
                yaw0=ph.yaw_start,
                yaw1=ph.yaw_goal,
                swing_goal_world=swing_goal_world,
                stance_anchor_world=stance_anchor_world,
            )

            interval_steps = max(1, int(self.cfg.phase_steps))
            phase_duration = max(dt, float(interval_steps) * dt)
            swing_duration = phase_duration
            if bool(self.cfg.swing_use_stance_gate):
                stance_phase = float(np.clip(self.cfg.swing_stance_phase, 0.0, 0.95))
                swing_duration = phase_duration * max(1.0 - stance_phase, 1e-6)

            touchdown_counter = {leg: 0 for leg in swing_legs}
            touchdown_latched = {leg: False for leg in swing_legs}
            hold_counter = 0
            timeout_counter = 0
            interval_timed_out = False
            touchdown_any = False
            swing_ground_contact_any = False

            for sub_global in range(interval_steps):
                alpha_global = float(sub_global + 1) / float(max(1, interval_steps))
                alpha_local = alpha_global
                alpha_s = self._swing_time_from_phase(alpha_local)

                edge_s = _smoothstep(alpha_global)
                base_xy_ref = (1.0 - edge_s) * phase.start_mid_xy + edge_s * phase.goal_mid_xy
                base_yaw_ref = _interp_angle(phase.start_yaw, phase.goal_yaw, edge_s)

                target_feet = self._build_foot_targets(phase, alpha_s, phase_duration=swing_duration)
                foot_vel_ref = self._build_foot_velocity_refs(phase, alpha_s, phase_duration=swing_duration)

                support_contacts = len(self._ground_contact_legs())
                base_wrench: Optional[Tuple[np.ndarray, np.ndarray]] = None
                if self.cfg.use_base_pd:
                    base_wrench = self._apply_base_pd(base_xy_ref, base_yaw_ref, support_contacts)
                else:
                    self.data.qfrc_applied[:] = 0.0

                ctrl = self._compose_leg_torques(
                    phase=phase,
                    target_feet=target_feet,
                    foot_vel_ref=foot_vel_ref,
                    base_wrench=base_wrench,
                )
                self.data.ctrl[:] = ctrl
                self.mujoco.mj_step(self.model, self.data)
                self._update_contact_measurements()
                ctrl_hist.append(ctrl.copy())
                for leg in LEG_ORDER:
                    if leg in phase.swing_legs:
                        swing_ref_hist[leg].append(np.asarray(target_feet[leg], dtype=np.float64).copy())
                    else:
                        swing_ref_hist[leg].append(np.array([np.nan, np.nan, np.nan], dtype=np.float64))

                for leg in phase.swing_legs:
                    if touchdown_latched[leg]:
                        continue
                    if self._strong_touchdown_ready(leg):
                        touchdown_latched[leg] = True
                        touchdown_counter[leg] = int(self.cfg.touchdown_stable_steps)
                    elif self._touchdown_candidate_ready(leg, target_feet[leg]):
                        touchdown_counter[leg] += 1
                    else:
                        if leg in self._ground_contact_legs():
                            touchdown_counter[leg] = max(
                                0, touchdown_counter[leg] - int(self.cfg.touchdown_counter_decay_on_contact)
                            )
                        else:
                            touchdown_counter[leg] = 0
                    if touchdown_counter[leg] >= int(self.cfg.touchdown_stable_steps):
                        touchdown_latched[leg] = True

                touchdown_all = all(
                    touchdown_latched[leg] or touchdown_counter[leg] >= int(self.cfg.touchdown_stable_steps)
                    for leg in phase.swing_legs
                ) if phase.swing_legs else True
                touchdown_any = touchdown_any or any(bool(v) for v in touchdown_latched.values())

                rpy = _quat_to_rpy_wxyz(self.data.qpos[3:7])
                contacts = self._ground_contact_legs()
                swing_ground_contact_any = swing_ground_contact_any or any(leg in contacts for leg in phase.swing_legs)
                swing_ground_contact_all = all(leg in contacts for leg in phase.swing_legs) if phase.swing_legs else True
                hold_ready = bool(touchdown_all)
                if bool(self.cfg.allow_early_switch_on_contact):
                    hold_ready = hold_ready or (
                        bool(swing_ground_contact_all)
                        and alpha_local >= float(np.clip(self.cfg.early_switch_contact_alpha_min, 0.0, 1.0))
                    )
                if hold_ready:
                    hold_counter += 1
                else:
                    hold_counter = 0

                qpos_hist.append(self.data.qpos.copy())
                qvel_hist.append(self.data.qvel.copy())
                rpy_hist.append(rpy.copy())
                contact_hist.append(contacts)
                debug_hist.append(
                    {
                        "segment": int(step_idx),
                        "interval": int(step_idx),
                        "substep": int(sub_global),
                        "phase_alpha": alpha_global,
                        "swing_legs": phase.swing_legs,
                        "stance_legs": phase.stance_legs,
                        "base_xy_ref": base_xy_ref.copy(),
                        "base_yaw_ref": base_yaw_ref,
                        "base_rpy": rpy.copy(),
                        "normal_force": {leg: float(self._normal_force_by_leg[leg]) for leg in LEG_ORDER},
                        "contact_surface": {leg: str(self._contact_surface_by_leg[leg]) for leg in LEG_ORDER},
                        "touchdown_latched": touchdown_latched.copy(),
                        "touchdown_timeout_counter": int(timeout_counter),
                        "governed": True,
                    }
                )

                if support_contacts < int(self.cfg.base_support_contact_min):
                    low_support_counter += 1
                else:
                    low_support_counter = 0
                if low_support_counter >= int(max(1, self.cfg.support_contact_abort_steps)):
                    terminated = True
                    term_reason = "support_contact_abort"
                    break

                if np.max(np.abs(rpy[:2])) > float(self.cfg.roll_pitch_abort_rad):
                    terminated = True
                    term_reason = "roll_pitch_abort"
                    break

                if not touchdown_all:
                    if alpha_local >= float(np.clip(self.cfg.touchdown_timeout_alpha, 0.0, 1.0)):
                        timeout_counter += 1
                    else:
                        timeout_counter = 0
                else:
                    timeout_counter = 0
                # End-of-phase backstop: in the governed path each step is exactly
                # ``phase_steps`` substeps, so the alpha-tail counter above can rarely reach
                # ``touchdown_timeout_steps`` (unlike follow_plan, whose multi-edge intervals
                # are long).  Use a CONTACT-based success contract: a step only "times out"
                # if the swing foot made NO ground contact at all during the step (a genuine
                # whiff -- e.g. the multi-metre raw swing this fix targets).  A foot that
                # contacted but did not complete the strict force/tangent/xy latch is a
                # successful step (the base-PD-driven body keeps advancing upright), matching
                # the permissive-touchdown contract the flat-straight case relied on.
                end_of_phase_timeout = (
                    sub_global == interval_steps - 1
                    and not touchdown_all
                    and not swing_ground_contact_any
                )
                if timeout_counter >= int(max(1, self.cfg.touchdown_timeout_steps)) or end_of_phase_timeout:
                    interval_timed_out = True
                    if bool(self.cfg.force_interval_end_on_touchdown_timeout):
                        break
                    terminated = True
                    term_reason = "touchdown_timeout_abort"
                    break

                if (
                    bool(self.cfg.allow_early_switch)
                    and bool(touchdown_all)
                    and hold_counter >= int(self.cfg.post_touchdown_hold_steps)
                ):
                    break

            # Commit: freeze stance anchors of the legs that just swung to their governed
            # stone targets (executed-foothold reference), regardless of touchdown quality.
            for leg in swing_legs:
                landed = self._current_foot_pos(leg)
                executed_anchor[leg] = phase.swing_goal[leg].copy()
                executed_footholds.append(
                    {
                        "step": int(step_idx),
                        "leg": str(leg),
                        "target_xy": [float(phase.swing_goal[leg][0]), float(phase.swing_goal[leg][1])],
                        "landed_xy": [float(landed[0]), float(landed[1])],
                        "landed_z": float(landed[2]),
                        "track_err_xy": float(np.linalg.norm(landed[:2] - phase.swing_goal[leg][:2])),
                        "touchdown": bool(touchdown_latched.get(leg, False)),
                    }
                )

            step_stats.append(
                {
                    "step": int(step_idx),
                    "swing_legs": list(swing_legs),
                    "touchdown_any": bool(touchdown_any),
                    "swing_contact_any": bool(swing_ground_contact_any),
                    "timed_out": bool(interval_timed_out),
                }
            )
            if interval_timed_out:
                timeout_streak += 1
            else:
                timeout_streak = 0
            if timeout_streak >= int(max(1, self.cfg.abort_after_consecutive_timeouts)):
                terminated = True
                term_reason = "consecutive_touchdown_timeouts"

            if terminated:
                break

        # Goal-hold: once the gait completes, hold the final pose (all-stance, base PD to
        # the final goal) for a few steps so the base-PD-driven body settles onto the goal
        # instead of stopping short by its tracking lag.
        if not terminated and int(self.cfg.goal_hold_steps) > 0 and len(phases) > 0:
            pf = phases[-1]
            goal_hold_phase = self._make_phase_state_governed(
                swing_legs=(), stance_legs=LEG_ORDER,
                mid0=pf.body_mid_goal, mid1=pf.body_mid_goal,
                yaw0=pf.yaw_goal, yaw1=pf.yaw_goal,
                swing_goal_world={},
                stance_anchor_world={leg: executed_anchor[leg] for leg in LEG_ORDER},
            )
            hold_dur = max(dt, float(self.cfg.goal_hold_steps) * dt)
            goal_mid = np.asarray(pf.body_mid_goal, dtype=np.float64).reshape(2)
            for _ in range(int(self.cfg.goal_hold_steps)):
                support_contacts = len(self._ground_contact_legs())
                if self.cfg.use_base_pd:
                    base_wrench = self._apply_base_pd(goal_mid, float(pf.yaw_goal), support_contacts)
                else:
                    self.data.qfrc_applied[:] = 0.0
                    base_wrench = None
                tf = self._build_foot_targets(goal_hold_phase, 1.0, phase_duration=hold_dur)
                fvr = self._build_foot_velocity_refs(goal_hold_phase, 1.0, phase_duration=hold_dur)
                ctrl = self._compose_leg_torques(phase=goal_hold_phase, target_feet=tf,
                                                 foot_vel_ref=fvr, base_wrench=base_wrench)
                self.data.ctrl[:] = ctrl
                self.mujoco.mj_step(self.model, self.data)
                self._update_contact_measurements()
                ctrl_hist.append(ctrl.copy())
                qpos_hist.append(self.data.qpos.copy())
                qvel_hist.append(self.data.qvel.copy())
                rpy_hist.append(_quat_to_rpy_wxyz(self.data.qpos[3:7]))
                contact_hist.append(self._ground_contact_legs())
                for leg in LEG_ORDER:
                    swing_ref_hist[leg].append(np.array([np.nan, np.nan, np.nan], dtype=np.float64))

        goal_xy = np.asarray(phases[-1].body_mid_goal, dtype=np.float64).reshape(2)
        if len(qpos_hist) > 0:
            base_xy_end = np.asarray(qpos_hist[-1][:2], dtype=np.float64)
            base_xy_start = np.asarray(qpos_hist[0][:2], dtype=np.float64)
            goal_error_xy = float(np.linalg.norm(base_xy_end - goal_xy))
            progress_xy = float(np.linalg.norm(base_xy_end - base_xy_start))
        else:
            goal_error_xy = float("nan")
            progress_xy = 0.0
        rpy_arr = np.asarray(rpy_hist, dtype=np.float64) if len(rpy_hist) > 0 else np.zeros((0, 3), dtype=np.float64)
        step_timeout_ratio = (
            float(np.mean([1.0 if bool(x["timed_out"]) else 0.0 for x in step_stats])) if step_stats else 0.0
        )
        n_steps_completed = len(step_stats)
        summary = {
            "goal_error_xy": goal_error_xy,
            "base_progress_xy": progress_xy,
            "pitch_abs_max": float(np.max(np.abs(rpy_arr[:, 1]))) if rpy_arr.size > 0 else 0.0,
            "roll_abs_max": float(np.max(np.abs(rpy_arr[:, 0]))) if rpy_arr.size > 0 else 0.0,
            "interval_count": int(n_steps_completed),
            "interval_timeout_ratio": float(step_timeout_ratio),
            "touchdown_any_ratio": float(np.mean([1.0 if bool(x["touchdown_any"]) else 0.0 for x in step_stats]))
            if step_stats else 0.0,
            "swing_contact_any_ratio": float(np.mean([1.0 if bool(x["swing_contact_any"]) else 0.0 for x in step_stats]))
            if step_stats else 0.0,
            "steps_completed": int(n_steps_completed),
            "steps_planned": int(len(phases)),
        }

        return {
            "qpos": np.asarray(qpos_hist),
            "qvel": np.asarray(qvel_hist),
            "ctrl": np.asarray(ctrl_hist),
            "base_rpy": np.asarray(rpy_hist),
            "contact_legs": contact_hist,
            "debug": debug_hist,
            "swing_ref": {leg: np.asarray(vals, dtype=np.float64) for leg, vals in swing_ref_hist.items()},
            "interval_stats": step_stats,
            "summary": summary,
            "terminated": terminated,
            "termination_reason": term_reason,
            "governed": {
                "used": True,
                "gait": str(getattr(gait_ref, "gait", "walk")),
                "n_steps": int(len(phases)),
                "steps_completed": int(n_steps_completed),
                "step_stats": step_stats,
                "executed_footholds": executed_footholds,
                "diagnostics": dict(getattr(gait_ref, "diagnostics", {}) or {}),
                "max_swing_distance": float(getattr(gait_ref, "max_swing_distance", 0.0)),
                "source_candidate_idx": int(getattr(gait_ref, "source_candidate_idx", -1)),
            },
        }

    # ------------------------------------------------------------------
    # Setup / indices
    # ------------------------------------------------------------------
    def _build_indices(self) -> None:
        mujoco = self.mujoco
        m = self.model

        self.leg_qidx: Dict[str, np.ndarray] = {}
        self.leg_didx: Dict[str, np.ndarray] = {}
        self.leg_geom_id: Dict[str, int] = {}
        self.leg_body_id: Dict[str, int] = {}
        self.joint_qidx_all: List[int] = []
        self.joint_didx_all: List[int] = []
        self.actuator_ids: List[int] = []
        self.support_geom_ids: set[int] = set()
        self.world_geom_ids: set[int] = set()
        self.geom_surface_type: Dict[int, str] = {}

        for leg, joints in LEG_JOINTS.items():
            qids, dids = [], []
            for jn in joints:
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid < 0:
                    raise KeyError(f"Missing joint: {jn}")
                qids.append(int(m.jnt_qposadr[jid]))
                dids.append(int(m.jnt_dofadr[jid]))

            self.leg_qidx[leg] = np.asarray(qids, dtype=np.int32)
            self.leg_didx[leg] = np.asarray(dids, dtype=np.int32)
            self.joint_qidx_all.extend(qids)
            self.joint_didx_all.extend(dids)

            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, LEG_TIP_GEOM[leg])
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, LEG_TIP_BODY[leg])
            if gid < 0 or bid < 0:
                raise KeyError(f"Missing foot geom/body for leg {leg}")
            self.leg_geom_id[leg] = int(gid)
            self.leg_body_id[leg] = int(bid)

            for actuator_name in (f"{leg}_hip", f"{leg}_thigh", f"{leg}_calf"):
                aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                if aid < 0:
                    raise KeyError(f"Missing actuator: {actuator_name}")
                self.actuator_ids.append(int(aid))

        self.joint_qidx_all = np.asarray(self.joint_qidx_all, dtype=np.int32)
        self.joint_didx_all = np.asarray(self.joint_didx_all, dtype=np.int32)
        self.actuator_ids = np.asarray(self.actuator_ids, dtype=np.int32)

        for gid in range(int(m.ngeom)):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name == "floor":
                self.support_geom_ids.add(int(gid))
                self.world_geom_ids.add(int(gid))
                self.geom_surface_type[int(gid)] = "bank"
            elif name.startswith("stepping_stone_") or name.startswith("bank_"):
                self.support_geom_ids.add(int(gid))
                self.world_geom_ids.add(int(gid))
                self.geom_surface_type[int(gid)] = "stone" if name.startswith("stepping_stone_") else "bank"
            elif name == "river_bottom":
                self.world_geom_ids.add(int(gid))
                self.geom_surface_type[int(gid)] = "river"

    # ------------------------------------------------------------------
    # Plan decoding
    # ------------------------------------------------------------------
    def _decode_plan(
        self,
        plan_states: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, np.ndarray]], np.ndarray, np.ndarray]:
        ps = np.asarray(plan_states, dtype=np.float64)
        mids_xy, yaws, feet, phase = decode_plan_states(
            plan_states,
            step_width=float(self.cfg.step_width),
            half_pair_length=float(self.cfg.leg_half_length),
            centerline_y=float(self.cfg.centerline_y),
            x_f_nominal=float(self.cfg.x_f_nominal),
            x_r_nominal=float(self.cfg.x_r_nominal),
            y_L_nominal=float(self.cfg.y_L_nominal),
            y_R_nominal=float(self.cfg.y_R_nominal),
        )
        mids_xy = np.asarray(mids_xy, dtype=np.float64)
        yaws = np.asarray(yaws, dtype=np.float64).reshape(-1)
        leg_goals: List[Dict[str, np.ndarray]] = []
        n = int(mids_xy.shape[0])
        for i in range(n):
            phase_goals: Dict[str, np.ndarray] = {}
            for leg in LEG_ORDER:
                goal = np.asarray(feet[leg][i], dtype=np.float64).copy()
                if goal.shape[0] == 2:
                    goal = np.array([goal[0], goal[1], self._terrain_height_at(goal[:2])], dtype=np.float64)
                # Keep planner XY as-is even if unsafe; only use terrain-aware Z.
                goal[2] = max(float(self.cfg.min_foot_z), float(goal[2]))
                phase_goals[leg] = goal
            leg_goals.append(phase_goals)
        modes = np.asarray(phase, dtype=np.int32).reshape(-1)
        taus = np.zeros((n,), dtype=np.float64)
        if ps.ndim == 2 and ps.shape[0] == n:
            if ps.shape[1] >= 16:
                modes = np.asarray(np.rint(ps[:, 14]), dtype=np.int32).reshape(-1)
                taus = np.asarray(ps[:, 15], dtype=np.float64).reshape(-1)
            elif ps.shape[1] >= 12:
                modes = np.asarray(np.rint(ps[:, 10]), dtype=np.int32).reshape(-1)
                taus = np.asarray(ps[:, 11], dtype=np.float64).reshape(-1)
        if modes.shape[0] != n:
            modes = np.zeros((n,), dtype=np.int32)
        modes = np.mod(modes, 4)
        taus = np.clip(taus, 0.0, 1.0)
        return mids_xy, yaws, leg_goals, modes, taus

    # ------------------------------------------------------------------
    # Phase construction
    # ------------------------------------------------------------------
    def _make_phase_state(
        self,
        *,
        swing_legs: Tuple[str, ...],
        mid0: np.ndarray,
        mid1: np.ndarray,
        yaw0: float,
        yaw1: float,
        leg_goals0: Dict[str, np.ndarray],
        leg_goals1: Dict[str, np.ndarray],
        clearance_extra: float = 0.0,
    ) -> PhaseState:
        stance_legs = tuple(leg for leg in LEG_ORDER if leg not in swing_legs)
        swing_start = {leg: self._current_foot_pos(leg) for leg in swing_legs}
        swing_goal = {leg: np.asarray(leg_goals1[leg], dtype=np.float64).copy() for leg in swing_legs}
        swing_coeff: Dict[str, np.ndarray] = {}
        stance_anchor: Dict[str, np.ndarray] = {}
        blend = float(np.clip(self.cfg.stance_plan_blend, 0.0, 1.0))
        for leg in stance_legs:
            p_curr = np.asarray(self._current_foot_pos(leg), dtype=np.float64)
            p_plan = np.asarray(leg_goals0[leg], dtype=np.float64)
            stance_anchor[leg] = self._stance_anchor_target((1.0 - blend) * p_curr + blend * p_plan)

        for leg in stance_legs:
            stance_anchor[leg] = self._stance_anchor_target(stance_anchor[leg])
        for leg in swing_legs:
            swing_goal[leg][:2] = np.asarray(swing_goal[leg][:2], dtype=np.float64)
            swing_goal[leg][2] = float(self._terrain_height_at(swing_goal[leg][:2]))
            swing_goal[leg][2] = max(float(self.cfg.min_foot_z), float(swing_goal[leg][2]))
            apex_z = max(
                float(self.cfg.min_foot_z),
                float(max(swing_start[leg][2], swing_goal[leg][2]) + max(self.cfg.swing_height, float(clearance_extra))),
            )
            swing_coeff[leg] = self._swing_poly_coeff(
                swing_start[leg],
                swing_goal[leg],
                apex_z=float(apex_z),
            )

        return PhaseState(
            swing_legs=swing_legs,
            stance_legs=stance_legs,
            start_mid_xy=np.asarray(mid0, dtype=np.float64).copy(),
            goal_mid_xy=np.asarray(mid1, dtype=np.float64).copy(),
            start_yaw=float(yaw0),
            goal_yaw=float(yaw1),
            swing_start=swing_start,
            swing_goal=swing_goal,
            swing_coeff=swing_coeff,
            stance_anchor=stance_anchor,
        )

    def _build_foot_targets(
        self,
        phase: PhaseState,
        alpha: float,
        *,
        phase_duration: float,
    ) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for leg in phase.stance_legs:
            out[leg] = phase.stance_anchor[leg].copy()
        for leg in phase.swing_legs:
            p, _v = self._swing_target(
                phase.swing_coeff[leg],
                alpha,
                phase_duration=phase_duration,
            )
            out[leg] = p
        return out

    def _build_foot_velocity_refs(
        self,
        phase: PhaseState,
        alpha: float,
        *,
        phase_duration: float,
    ) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for leg in phase.stance_legs:
            out[leg] = np.zeros(3, dtype=np.float64)
        for leg in phase.swing_legs:
            _, v = self._swing_target(
                phase.swing_coeff[leg],
                float(alpha),
                phase_duration=phase_duration,
            )
            out[leg] = v
        return out

    def _swing_poly_coeff(self, start: np.ndarray, land_goal: np.ndarray, *, apex_z: float) -> np.ndarray:
        p0 = np.asarray(start, dtype=np.float64).reshape(3)
        pf = np.asarray(land_goal, dtype=np.float64).reshape(3)
        pc = 0.5 * (p0 + pf)
        pc[2] = max(float(self.cfg.min_foot_z), float(apex_z))
        b = np.vstack(
            [
                p0,
                pf,
                pc,
                np.zeros((1, 3), dtype=np.float64),
                np.zeros((1, 3), dtype=np.float64),
                np.zeros((1, 3), dtype=np.float64),
                np.zeros((1, 3), dtype=np.float64),
            ]
        )
        return (_SEXTIC_A_INV @ b).astype(np.float64)

    def _swing_target(
        self,
        coeff: np.ndarray,
        alpha: float,
        *,
        phase_duration: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        a = float(np.clip(alpha, 0.0, 1.0))
        coeff = np.asarray(coeff, dtype=np.float64).reshape(7, 3)
        v_pos = np.array([1.0, a, a * a, a**3, a**4, a**5, a**6], dtype=np.float64)
        v_vel = np.array([0.0, 1.0, 2.0 * a, 3.0 * a * a, 4.0 * a**3, 5.0 * a**4, 6.0 * a**5], dtype=np.float64)
        pos = v_pos @ coeff
        vel_unit = v_vel @ coeff
        vel = vel_unit / max(float(phase_duration), 1e-6)
        pos[2] = max(float(self.cfg.min_foot_z), float(pos[2]))
        return pos.astype(np.float64), vel.astype(np.float64)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def _compose_leg_torques(
        self,
        *,
        phase: PhaseState,
        target_feet: Dict[str, np.ndarray],
        foot_vel_ref: Dict[str, np.ndarray],
        base_wrench: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        tau = np.zeros((len(self.actuator_ids),), dtype=np.float64)
        leg_to_slice = {
            "FL": slice(0, 3),
            "FR": slice(3, 6),
            "RL": slice(6, 9),
            "RR": slice(9, 12),
        }

        for leg in phase.stance_legs:
            kp_xy = float(self.cfg.stance_kp_xy)
            kp_z = float(self.cfg.stance_kp_z)
            kd_xy = float(self.cfg.stance_kd_xy)
            kd_z = float(self.cfg.stance_kd_z)
            clip_xy = float(self.cfg.max_anchor_error_xy)
            if bool(self.cfg.flat_only_foot_lock) and bool(self._flat_ground_scene):
                kp_xy *= float(max(1.0, self.cfg.flat_lock_stance_kp_xy_scale))
                kp_z *= float(max(1.0, self.cfg.flat_lock_stance_kp_z_scale))
                kd_xy *= float(max(1.0, self.cfg.flat_lock_stance_kd_xy_scale))
                kd_z *= float(max(1.0, self.cfg.flat_lock_stance_kd_z_scale))
                clip_xy = float(max(clip_xy, self.cfg.flat_lock_max_anchor_error_xy))
            tau_leg = self._task_space_leg_torque(
                leg,
                target_feet[leg],
                foot_vel_ref[leg],
                kp_xy=kp_xy,
                kp_z=kp_z,
                kd_xy=kd_xy,
                kd_z=kd_z,
                clip_xy=clip_xy,
            )
            tau[leg_to_slice[leg]] += tau_leg

        if (
            bool(self.cfg.use_stance_force_distribution)
            and base_wrench is not None
            and len(phase.stance_legs) > 0
        ):
            fd_tau = self._stance_force_distribution_torques(
                phase.stance_legs,
                force_world=np.asarray(base_wrench[0], dtype=np.float64),
                torque_world=np.asarray(base_wrench[1], dtype=np.float64),
                leg_to_slice=leg_to_slice,
            )
            bfd = float(np.clip(self.cfg.stance_fd_torque_blend, 0.0, 1.0))
            tau += bfd * fd_tau

        for leg in phase.swing_legs:
            if bool(self.cfg.use_swing_jointspace_tracking):
                tau_js = self._swing_jointspace_leg_torque(
                    leg,
                    target_feet[leg],
                    foot_vel_ref[leg],
                )
                tau_ts = self._task_space_leg_torque(
                    leg,
                    target_feet[leg],
                    foot_vel_ref[leg],
                    kp_xy=float(self.cfg.swing_kp_xy),
                    kp_z=float(self.cfg.swing_kp_z),
                    kd_xy=float(self.cfg.swing_kd_xy),
                    kd_z=float(self.cfg.swing_kd_z),
                    clip_xy=float(self.cfg.max_swing_tracking_error_xy),
                )
                b = float(np.clip(self.cfg.swing_jointspace_tau_blend, 0.0, 1.0))
                tau_leg = (1.0 - b) * tau_ts + b * tau_js
            else:
                tau_leg = self._task_space_leg_torque(
                    leg,
                    target_feet[leg],
                    foot_vel_ref[leg],
                    kp_xy=float(self.cfg.swing_kp_xy),
                    kp_z=float(self.cfg.swing_kp_z),
                    kd_xy=float(self.cfg.swing_kd_xy),
                    kd_z=float(self.cfg.swing_kd_z),
                    clip_xy=float(self.cfg.max_swing_tracking_error_xy),
                )
            tau[leg_to_slice[leg]] += tau_leg

        q_joint = np.asarray(self.data.qpos[self.joint_qidx_all], dtype=np.float64)
        qd_joint = np.asarray(self.data.qvel[self.joint_didx_all], dtype=np.float64)
        q_ref = np.asarray(self._home_qpos[self.joint_qidx_all], dtype=np.float64)
        tau_reg = float(self.cfg.joint_kp) * (q_ref - q_joint) - float(self.cfg.joint_kd) * qd_joint
        tau += np.clip(tau_reg, -float(self.cfg.joint_torque_clip), float(self.cfg.joint_torque_clip))

        return np.clip(tau, -float(self.cfg.task_torque_clip), float(self.cfg.task_torque_clip))

    def _apply_base_pd(
        self,
        base_xy: np.ndarray,
        base_yaw: float,
        support_contacts: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        q = self.data.qpos
        v = self.data.qvel
        rpy = _quat_to_rpy_wxyz(q[3:7])
        yaw_err = _wrap_angle(float(base_yaw) - float(rpy[2]))

        if support_contacts < int(cfg.base_support_contact_min):
            support_scale = 0.0
        else:
            support_scale = float(np.clip(float(support_contacts) / 4.0, 0.0, 1.0))
        support_scale = float(max(support_scale, float(np.clip(cfg.base_support_scale_min, 0.0, 1.0))))

        fx = float(cfg.base_kp_xy) * (float(base_xy[0]) - float(q[0])) - float(cfg.base_kd_xy) * float(v[0])
        fy = float(cfg.base_kp_xy) * (float(base_xy[1]) - float(q[1])) - float(cfg.base_kd_xy) * float(v[1])
        z_ref_nom = self._base_height + float(cfg.base_height_offset)
        if bool(cfg.use_dynamic_base_z_ref):
            foot_z = np.zeros((len(LEG_ORDER),), dtype=np.float64)
            for i, leg in enumerate(LEG_ORDER):
                foot_z[i] = float(self._current_foot_pos(leg)[2])
            z_ref_dyn = float(np.mean(foot_z) + float(cfg.base_target_clearance_from_feet))
            z_ref = max(float(cfg.base_z_min), z_ref_nom, z_ref_dyn)
        else:
            z_ref = z_ref_nom
        gmag = float(abs(self.model.opt.gravity[2]))
        fz = (
            float(cfg.base_weight_comp) * self._total_mass * gmag
            + float(cfg.base_kp_z) * (z_ref - float(q[2]))
            - float(cfg.base_kd_z) * float(v[2])
        )
        tx = -float(cfg.base_kp_rp) * float(rpy[0]) - float(cfg.base_kd_rp) * float(v[3])
        ty = -float(cfg.base_kp_rp) * float(rpy[1]) - float(cfg.base_kd_rp) * float(v[4])
        tz = float(cfg.base_kp_yaw) * yaw_err - float(cfg.base_kd_yaw) * float(v[5])

        force_world = np.array(
            [
                support_scale * np.clip(fx, -cfg.base_force_xy_clip, cfg.base_force_xy_clip),
                support_scale * np.clip(fy, -cfg.base_force_xy_clip, cfg.base_force_xy_clip),
                support_scale * np.clip(fz, -cfg.base_force_z_clip, cfg.base_force_z_clip),
            ],
            dtype=np.float64,
        )
        torque_world = np.array(
            [
                support_scale * np.clip(tx, -cfg.base_torque_clip, cfg.base_torque_clip),
                support_scale * np.clip(ty, -cfg.base_torque_clip, cfg.base_torque_clip),
                support_scale * np.clip(tz, -cfg.base_torque_clip, cfg.base_torque_clip),
            ],
            dtype=np.float64,
        )
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[0:3] = force_world
        self.data.qfrc_applied[3:6] = torque_world
        return force_world, torque_world

    def _stance_force_distribution_torques(
        self,
        stance_legs: Tuple[str, ...],
        *,
        force_world: np.ndarray,
        torque_world: np.ndarray,
        leg_to_slice: Dict[str, slice],
    ) -> np.ndarray:
        n = len(stance_legs)
        tau = np.zeros((len(self.actuator_ids),), dtype=np.float64)
        if n <= 0:
            return tau
        A = np.zeros((6, 3 * n), dtype=np.float64)
        com = np.asarray(self.data.qpos[:3], dtype=np.float64)
        for i, leg in enumerate(stance_legs):
            p = self._current_foot_pos(leg)
            r = np.asarray(p - com, dtype=np.float64)
            A[0:3, 3 * i : 3 * i + 3] = np.eye(3, dtype=np.float64)
            rx = np.array(
                [[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]],
                dtype=np.float64,
            )
            A[3:6, 3 * i : 3 * i + 3] = rx
        b = np.concatenate([np.asarray(force_world, dtype=np.float64), np.asarray(torque_world, dtype=np.float64)])
        x = self._solve_stance_qp(A, b)
        if x is None:
            lam = float(max(1e-8, self.cfg.stance_fd_lambda))
            x = np.linalg.solve(A.T @ A + lam * np.eye(3 * n, dtype=np.float64), A.T @ b)

        mu = float(max(0.0, self.cfg.stance_fd_mu))
        fz_min = float(max(0.0, self.cfg.stance_fd_fz_min))
        fz_max = float(max(fz_min, self.cfg.stance_fd_fz_max))
        for i, leg in enumerate(stance_legs):
            f = np.asarray(x[3 * i : 3 * i + 3], dtype=np.float64)
            fz = float(np.clip(f[2], fz_min, fz_max))
            ft = np.asarray(f[:2], dtype=np.float64)
            ft_n = float(np.linalg.norm(ft))
            ft_max = float(max(0.0, mu * fz))
            if ft_n > ft_max and ft_n > 1e-9:
                ft = ft * (ft_max / ft_n)
            f = np.array([ft[0], ft[1], fz], dtype=np.float64)
            _p, _v, J = self._foot_pos_vel_jac_leg(leg)
            tau_leg = -J.T @ f
            tau[leg_to_slice[leg]] += tau_leg
        return np.clip(tau, -float(self.cfg.task_torque_clip), float(self.cfg.task_torque_clip))

    def _solve_stance_qp(self, A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
        if not bool(self.cfg.use_stance_qp):
            return None
        n = int(A.shape[1])
        if n <= 0:
            return None
        try:
            from qpsolvers import solve_qp
        except Exception:
            return None

        lam = float(max(1e-8, self.cfg.stance_fd_lambda))
        P = A.T @ A + lam * np.eye(n, dtype=np.float64)
        q = -A.T @ b

        n_legs = n // 3
        G = np.zeros((6 * n_legs, n), dtype=np.float64)
        h = np.zeros((6 * n_legs,), dtype=np.float64)
        mu = float(max(0.0, self.cfg.stance_fd_mu))
        fz_min = float(max(0.0, self.cfg.stance_fd_fz_min))
        fz_max = float(max(fz_min, self.cfg.stance_fd_fz_max))
        for i in range(n_legs):
            c = 3 * i
            r = 6 * i
            G[r + 0, c + 0] = 1.0
            G[r + 0, c + 2] = -mu
            G[r + 1, c + 0] = -1.0
            G[r + 1, c + 2] = -mu
            G[r + 2, c + 1] = 1.0
            G[r + 2, c + 2] = -mu
            G[r + 3, c + 1] = -1.0
            G[r + 3, c + 2] = -mu
            G[r + 4, c + 2] = -1.0
            h[r + 4] = -fz_min
            G[r + 5, c + 2] = 1.0
            h[r + 5] = fz_max

        for solver_name in tuple(self.cfg.stance_qp_solver_order):
            try:
                sol = solve_qp(P, q, G, h, solver=str(solver_name))
            except Exception:
                sol = None
            if sol is not None and np.all(np.isfinite(sol)):
                return np.asarray(sol, dtype=np.float64)
        return None

    # ------------------------------------------------------------------
    # Foot kinematics / task-space torque
    # ------------------------------------------------------------------
    def _current_foot_pos(self, leg: str) -> np.ndarray:
        gid = self.leg_geom_id[leg]
        return np.asarray(self.data.geom_xpos[gid], dtype=np.float64).copy()

    def _foot_pos_vel_jac_leg(self, leg: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mujoco = self.mujoco
        gid = self.leg_geom_id[leg]
        didx = self.leg_didx[leg]
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacGeom(self.model, self.data, jacp, jacr, gid)
        J_leg = jacp[:, didx]
        pos = np.asarray(self.data.geom_xpos[gid], dtype=np.float64).copy()
        vel = J_leg @ np.asarray(self.data.qvel[didx], dtype=np.float64)
        return pos, vel, J_leg

    def _task_space_leg_torque(
        self,
        leg: str,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        *,
        kp_xy: float,
        kp_z: float,
        kd_xy: float,
        kd_z: float,
        clip_xy: float,
    ) -> np.ndarray:
        pos, vel, J = self._foot_pos_vel_jac_leg(leg)
        err = np.asarray(target_pos, dtype=np.float64) - pos
        err[0] = float(np.clip(err[0], -clip_xy, clip_xy))
        err[1] = float(np.clip(err[1], -clip_xy, clip_xy))
        vel_err = np.asarray(target_vel, dtype=np.float64) - vel

        wrench = np.array(
            [
                float(kp_xy) * err[0] + float(kd_xy) * vel_err[0],
                float(kp_xy) * err[1] + float(kd_xy) * vel_err[1],
                float(kp_z) * err[2] + float(kd_z) * vel_err[2],
            ],
            dtype=np.float64,
        )
        tau = J.T @ wrench
        return np.clip(tau, -float(self.cfg.task_torque_clip), float(self.cfg.task_torque_clip))

    def _swing_jointspace_leg_torque(
        self,
        leg: str,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
    ) -> np.ndarray:
        """Resolved-IK style swing tracking in joint space (quadruped_control-like)."""
        qidx = self.leg_qidx[leg]
        didx = self.leg_didx[leg]
        q = np.asarray(self.data.qpos[qidx], dtype=np.float64)
        qd = np.asarray(self.data.qvel[didx], dtype=np.float64)
        p, _v_curr, J = self._foot_pos_vel_jac_leg(leg)

        p_ref = np.asarray(target_pos[:3], dtype=np.float64)
        v_ref = np.asarray(target_vel[:3], dtype=np.float64)
        e = p_ref - p
        v_cmd = v_ref + float(self.cfg.swing_ik_kp_pos) * e

        JT = J.T
        lam = float(max(1e-6, self.cfg.swing_ik_damping))
        A = J @ JT + (lam * lam) * np.eye(3, dtype=np.float64)
        qd_ref = JT @ np.linalg.solve(A, v_cmd)
        qd_ref = np.clip(qd_ref, -float(self.cfg.swing_qd_ref_clip), float(self.cfg.swing_qd_ref_clip))

        dt = float(max(1e-6, self.cfg.sim_dt))
        q_ref = q + dt * qd_ref
        dq = np.clip(q_ref - q, -float(self.cfg.swing_q_step_clip), float(self.cfg.swing_q_step_clip))
        q_ref = q + dq

        tau = float(self.cfg.swing_joint_kp) * (q_ref - q) + float(self.cfg.swing_joint_kd) * (qd_ref - qd)
        return np.clip(tau, -float(self.cfg.task_torque_clip), float(self.cfg.task_torque_clip))

    # ------------------------------------------------------------------
    # Contact sensing (mj_contactForce version)
    # ------------------------------------------------------------------
    def _clear_contact_cache(self) -> None:
        self._contact_wrench_by_leg = {leg: np.zeros(6, dtype=np.float64) for leg in LEG_ORDER}
        self._normal_force_by_leg = {leg: 0.0 for leg in LEG_ORDER}
        self._tangent_speed_by_leg = {leg: 0.0 for leg in LEG_ORDER}
        self._ground_contact_set = set()
        self._contact_surface_by_leg = {leg: "none" for leg in LEG_ORDER}
        self._contact_surface_force_by_leg = {leg: 0.0 for leg in LEG_ORDER}

    def _leg_from_geom(self, gid: int) -> Optional[str]:
        for leg, foot_gid in self.leg_geom_id.items():
            if int(foot_gid) == int(gid):
                return leg
        return None

    def _update_contact_measurements(self) -> None:
        mujoco = self.mujoco
        self._clear_contact_cache()

        for i in range(int(self.data.ncon)):
            c = self.data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            leg1 = self._leg_from_geom(g1)
            leg2 = self._leg_from_geom(g2)

            leg = None
            other_gid = None
            sign = 1.0
            if leg1 is not None and g2 in self.world_geom_ids:
                leg = leg1
                other_gid = g2
                sign = 1.0
            elif leg2 is not None and g1 in self.world_geom_ids:
                leg = leg2
                other_gid = g1
                sign = -1.0
            else:
                continue

            wrench_contact = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, wrench_contact)
            # MuJoCo expresses force in the contact frame. We use the contact normal magnitude as f_n,
            # and combine the tangential components as f_t.
            force_contact = sign * wrench_contact[:3]
            self._contact_wrench_by_leg[leg] += sign * wrench_contact
            normal_f = max(0.0, float(force_contact[0]))
            self._normal_force_by_leg[leg] += normal_f
            self._ground_contact_set.add(leg)
            surface = str(self.geom_surface_type.get(int(other_gid), "none"))
            if normal_f >= float(self._contact_surface_force_by_leg.get(leg, 0.0)):
                self._contact_surface_force_by_leg[leg] = normal_f
                self._contact_surface_by_leg[leg] = surface

        for leg in LEG_ORDER:
            _, vel, _ = self._foot_pos_vel_jac_leg(leg)
            tangential_speed = float(np.linalg.norm(np.asarray(vel[:2], dtype=np.float64)))
            self._tangent_speed_by_leg[leg] = tangential_speed

    def _has_contact_with_geom_set(self, leg: str, geom_ids: Sequence[int]) -> bool:
        gid = self.leg_geom_id[leg]
        geom_ids = set(int(x) for x in geom_ids)
        for i in range(int(self.data.ncon)):
            c = self.data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            if (g1 == gid and g2 in geom_ids) or (g2 == gid and g1 in geom_ids):
                return True
        return False

    def _ground_contact_legs(self) -> List[str]:
        return [leg for leg in LEG_ORDER if leg in self._ground_contact_set]

    def _touchdown_candidate_ready(self, leg: str, target_foot: np.ndarray) -> bool:
        in_contact = leg in self._ground_contact_legs()
        surface = str(self._contact_surface_by_leg.get(leg, "none"))
        normal_ok = float(self._normal_force_by_leg[leg]) >= self._touchdown_force_threshold(surface, strong=False)
        tangent_ok = float(self._tangent_speed_by_leg[leg]) <= float(self.cfg.touchdown_tangent_speed_thresh)
        pos = self._current_foot_pos(leg)
        xy_ok = float(np.linalg.norm(pos[:2] - np.asarray(target_foot[:2], dtype=np.float64))) <= float(self.cfg.touchdown_xy_tol)
        return bool(in_contact and normal_ok and tangent_ok and xy_ok)

    def _strong_touchdown_ready(self, leg: str) -> bool:
        in_contact = leg in self._ground_contact_legs()
        surface = str(self._contact_surface_by_leg.get(leg, "none"))
        normal_ok = float(self._normal_force_by_leg[leg]) >= self._touchdown_force_threshold(surface, strong=True)
        tangent_ok = float(self._tangent_speed_by_leg[leg]) <= float(self.cfg.touchdown_tangent_speed_thresh)
        return bool(in_contact and normal_ok and tangent_ok)

    def _liftoff_ready(self, leg: str, current_target: np.ndarray) -> bool:
        # Optional helper if you later add an explicit liftoff gating stage.
        low_force = float(self._normal_force_by_leg[leg]) <= float(self.cfg.release_force_thresh)
        height_ok = float(self._current_foot_pos(leg)[2]) >= float(current_target[2]) + float(self.cfg.release_height_margin)
        return bool(low_force and height_ok)

    # ------------------------------------------------------------------
    # Terrain / safe projection
    # ------------------------------------------------------------------
    def _terrain_height_at(self, xy: np.ndarray) -> float:
        x = float(xy[0])
        y = float(xy[1])
        if not self.stepping_scene:
            return 0.0
        centers = np.asarray(self.stepping_scene.get("stones_centers", []), dtype=np.float64)
        radii = np.asarray(self.stepping_scene.get("stones_radii", []), dtype=np.float64).reshape(-1)
        n = min(len(centers), len(radii))
        for i in range(n):
            c = np.asarray(centers[i], dtype=np.float64).reshape(-1)
            if c.size < 2:
                continue
            if (x - c[0]) ** 2 + (y - c[1]) ** 2 <= float(radii[i]) ** 2:
                return float(self._scene_stone_top_z)
        if not bool(self.stepping_scene.get("has_river", True)):
            return 0.0
        river_x = self.stepping_scene.get("river_x", [-0.2, 0.2])
        try:
            rx0, rx1 = float(river_x[0]), float(river_x[1])
        except Exception:
            rx0, rx1 = -0.2, 0.2
        if rx0 <= x <= rx1:
            return -float(self._scene_river_depth)
        return float(self._scene_bank_top_z)

    def _project_to_safe_support(self, point: np.ndarray, *, margin: float = 0.006) -> np.ndarray:
        out = np.asarray(point, dtype=np.float64).copy()
        if not self.stepping_scene:
            return out

        centers = np.asarray(self.stepping_scene.get("stones_centers", []), dtype=np.float64)
        radii = np.asarray(self.stepping_scene.get("stones_radii", []), dtype=np.float64).reshape(-1)
        n = min(len(centers), len(radii))
        if n <= 0:
            return out

        centers = centers[:n, :2]
        radii = radii[:n]
        delta = out[:2][None, :] - centers
        dist = np.linalg.norm(delta, axis=1)
        support = radii - dist
        best_idx = int(np.argmax(support))
        if float(support[best_idx]) >= float(margin):
            return out

        direction = delta[best_idx]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            direction = np.array([1.0, 0.0], dtype=np.float64)
            norm = 1.0
        target_r = max(0.0, float(radii[best_idx]) - float(margin))
        out[:2] = centers[best_idx] + direction / norm * target_r
        out[2] = max(float(self.cfg.min_foot_z), float(self._scene_stone_top_z))
        return out

    def _stance_anchor_target(self, point: np.ndarray) -> np.ndarray:
        out = np.asarray(point, dtype=np.float64).copy()
        support_z = float(self._terrain_height_at(out[:2]))
        out[2] = max(
            float(self.cfg.min_foot_z),
            support_z + float(self.cfg.stance_press_z_offset),
        )
        return out

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _warm_start_stance(self, steps: int) -> None:
        for _ in range(max(0, int(steps))):
            self.data.ctrl[:] = 0.0
            self.data.qfrc_applied[:] = 0.0
            self.mujoco.mj_step(self.model, self.data)
            self._update_contact_measurements()

    def _set_base_pose(self, xy: np.ndarray, yaw: float) -> None:
        self.data.qpos[0] = float(xy[0])
        self.data.qpos[1] = float(xy[1])
        self.data.qpos[2] = max(float(self.cfg.base_z_min), self._base_height + float(self.cfg.base_height_offset))
        self.data.qpos[3:7] = _quat_wxyz_yaw(float(yaw))
        self.data.qvel[:6] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        self._update_contact_measurements()

    def _empty_rollout(self) -> Dict[str, Any]:
        return {
            "qpos": np.asarray([self.data.qpos.copy()]),
            "qvel": np.asarray([self.data.qvel.copy()]),
            "ctrl": np.zeros((0, int(self.model.nu)), dtype=np.float64),
            "base_rpy": np.asarray([_quat_to_rpy_wxyz(self.data.qpos[3:7])]),
            "contact_legs": [self._ground_contact_legs()],
            "debug": [],
            "swing_ref": {leg: np.zeros((0, 3), dtype=np.float64) for leg in LEG_ORDER},
            "interval_stats": [],
            "summary": {
                "goal_error_xy": float("nan"),
                "base_progress_xy": 0.0,
                "pitch_abs_max": 0.0,
                "roll_abs_max": 0.0,
                "interval_count": 0,
                "interval_timeout_ratio": 0.0,
                "touchdown_any_ratio": 0.0,
                "swing_contact_any_ratio": 0.0,
            },
            "terminated": False,
            "termination_reason": None,
        }


def load_stepping_plan_from_seed_dir(
    seed_dir: str | Path,
    *,
    step_width: float = 0.30,
    centerline_y: float = 0.0,
    x_f_nominal: float = 0.18,
    x_r_nominal: float = -0.18,
    y_L_nominal: float = 0.15,
    y_R_nominal: float = -0.15,
) -> Dict[str, Any]:
    import json

    seed_path = Path(seed_dir)
    traj_path = seed_path / "trajectory" / "trajectory.json"
    if not traj_path.exists():
        raise FileNotFoundError(traj_path)
    with open(traj_path, "r", encoding="utf-8") as f:
        blob = json.load(f) or {}
    cand = blob.get("candidate_states") or []
    if not cand:
        raise ValueError(f"No candidate_states in {traj_path}")
    idx = int(blob.get("best_idx", 0))
    idx = max(0, min(idx, len(cand) - 1))
    st = np.asarray(cand[idx], dtype=np.float32)
    if st.ndim != 2 or st.shape[1] < 3:
        raise ValueError(f"Unexpected state shape: {st.shape}")

    mid, yaw, feet, mode = decode_plan_states(
        st,
        step_width=float(step_width),
        half_pair_length=0.18,
        centerline_y=float(centerline_y),
        x_f_nominal=float(x_f_nominal),
        x_r_nominal=float(x_r_nominal),
        y_L_nominal=float(y_L_nominal),
        y_R_nominal=float(y_R_nominal),
    )
    if st.shape[1] >= 16:
        tau = st[:, 15].astype(np.float32)
    elif st.shape[1] >= 12:
        tau = st[:, 11].astype(np.float32)
    else:
        tau = np.zeros((st.shape[0],), dtype=np.float32)
    return {
        "states": st.astype(np.float32),
        "mid": np.asarray(mid, dtype=np.float32),
        "yaw": np.asarray(yaw, dtype=np.float32),
        "feet": {leg: np.asarray(feet[leg], dtype=np.float32) for leg in LEG_ORDER},
        "mode": np.asarray(mode, dtype=np.int32),
        "tau": np.clip(np.asarray(tau, dtype=np.float32), 0.0, 1.0),
    }
