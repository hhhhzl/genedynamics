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

LEG_JOINTS = {
    "FL": ("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"),
    "FR": ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"),
    "RL": ("RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"),
    "RR": ("RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"),
}
LEG_TIP_GEOM = {"FL": "FL", "FR": "FR", "RL": "RL", "RR": "RR"}
LEG_TIP_BODY = {"FL": "FL_calf", "FR": "FR_calf", "RL": "RL_calf", "RR": "RR_calf"}


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


@dataclass
class MinimalFollowerConfig:
    gait: str = "walk"  # walk | trot
    sim_dt: float = 0.01
    phase_steps: int = 36
    settle_steps: int = 30

    # Geometry / nominal posture
    step_width: float = 0.28
    leg_half_length: float = 0.18
    base_height_offset: float = 0.01
    base_z_min: float = 0.23
    min_foot_z: float = 0.015
    swing_height: float = 0.045

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

    # Foot task-space control
    swing_kp_xy: float = 170.0
    swing_kp_z: float = 230.0
    swing_kd_xy: float = 14.0
    swing_kd_z: float = 18.0
    stance_kp_xy: float = 110.0
    stance_kp_z: float = 135.0
    stance_kd_xy: float = 24.0
    stance_kd_z: float = 16.0

    # Joint regularization only
    joint_kp: float = 26.0
    joint_kd: float = 1.4
    joint_torque_clip: float = 3.0
    task_torque_clip: float = 12.0

    # Touchdown / release logic
    touchdown_force_thresh: float = 12.0
    touchdown_tangent_speed_thresh: float = 0.20
    touchdown_xy_tol: float = 0.06
    touchdown_stable_steps: int = 2
    post_touchdown_hold_steps: int = 8
    strong_touchdown_force_thresh: float = 28.0
    touchdown_counter_decay_on_contact: int = 1
    release_force_thresh: float = 5.0
    release_height_margin: float = 0.015
    release_stable_steps: int = 3
    allow_early_switch: bool = False

    # Safety / saturation
    roll_pitch_abort_rad: float = 0.45
    max_anchor_error_xy: float = 0.04
    max_swing_tracking_error_xy: float = 0.08
    swing_xy_finish_alpha: float = 0.80
    swing_z_only_tail: float = 0.20
    stance_press_z_offset: float = -0.004


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
    stance_anchor: Dict[str, np.ndarray]


class SteppingWalkFollowerMinimal:
    """
    Minimal follower, v2.

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

        self._contact_wrench_by_leg: Dict[str, np.ndarray] = {leg: np.zeros(6, dtype=np.float64) for leg in LEG_ORDER}
        self._normal_force_by_leg: Dict[str, float] = {leg: 0.0 for leg in LEG_ORDER}
        self._tangent_speed_by_leg: Dict[str, float] = {leg: 0.0 for leg in LEG_ORDER}

    # ------------------------------------------------------------------
    # Compatibility wrappers (closer to original style)
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
    def follow_plan(
        self,
        plan_states: np.ndarray,
        *,
        max_segments: Optional[int] = None,
    ) -> Dict[str, Any]:
        mids_xy, yaws, leg_goals = self._decode_plan(plan_states)
        n_seg = max(0, len(mids_xy) - 1)
        if max_segments is not None:
            n_seg = min(n_seg, int(max_segments))

        if n_seg <= 0:
            return self._empty_rollout()

        qpos_hist: List[np.ndarray] = []
        qvel_hist: List[np.ndarray] = []
        rpy_hist: List[np.ndarray] = []
        contact_hist: List[List[str]] = []
        debug_hist: List[Dict[str, Any]] = []

        self._set_base_pose(mids_xy[0], yaws[0])
        self._warm_start_stance(steps=self.cfg.settle_steps)

        phase_groups = WALK_GROUPS if self.cfg.gait == "walk" else TROT_GROUPS
        terminated = False
        term_reason = None

        for seg_idx in range(n_seg):
            swing_legs = phase_groups[seg_idx % len(phase_groups)]
            phase = self._make_phase_state(
                swing_legs=swing_legs,
                mid0=mids_xy[seg_idx],
                mid1=mids_xy[seg_idx + 1],
                yaw0=yaws[seg_idx],
                yaw1=yaws[seg_idx + 1],
                leg_goals0=leg_goals[seg_idx],
                leg_goals1=leg_goals[seg_idx + 1],
            )

            touchdown_counter = {leg: 0 for leg in phase.swing_legs}
            touchdown_latched = {leg: False for leg in phase.swing_legs}
            hold_counter = 0

            for sub in range(self.cfg.phase_steps):
                alpha = float(sub + 1) / float(max(1, self.cfg.phase_steps))
                alpha_s = _smoothstep(alpha)

                base_xy_ref = (1.0 - alpha_s) * phase.start_mid_xy + alpha_s * phase.goal_mid_xy
                base_yaw_ref = _interp_angle(phase.start_yaw, phase.goal_yaw, alpha_s)

                target_feet = self._build_foot_targets(phase, alpha_s)
                foot_vel_ref = self._build_foot_velocity_refs(phase, alpha_s)

                support_contacts = len(self._ground_contact_legs())
                if self.cfg.use_base_pd:
                    self._apply_base_pd(base_xy_ref, base_yaw_ref, support_contacts)
                else:
                    self.data.qfrc_applied[:] = 0.0

                ctrl = self._compose_leg_torques(
                    phase=phase,
                    target_feet=target_feet,
                    foot_vel_ref=foot_vel_ref,
                )
                self.data.ctrl[:] = ctrl
                self.mujoco.mj_step(self.model, self.data)
                self._update_contact_measurements()

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

                if touchdown_all:
                    hold_counter += 1
                else:
                    hold_counter = 0

                rpy = _quat_to_rpy_wxyz(self.data.qpos[3:7])
                contacts = self._ground_contact_legs()
                qpos_hist.append(self.data.qpos.copy())
                qvel_hist.append(self.data.qvel.copy())
                rpy_hist.append(rpy.copy())
                contact_hist.append(contacts)
                debug_hist.append(
                    {
                        "segment": seg_idx,
                        "substep": sub,
                        "phase_alpha": alpha_s,
                        "swing_legs": phase.swing_legs,
                        "stance_legs": phase.stance_legs,
                        "base_xy_ref": base_xy_ref.copy(),
                        "base_yaw_ref": base_yaw_ref,
                        "base_rpy": rpy.copy(),
                        "normal_force": {leg: float(self._normal_force_by_leg[leg]) for leg in LEG_ORDER},
                        "tangent_speed": {leg: float(self._tangent_speed_by_leg[leg]) for leg in LEG_ORDER},
                        "touchdown_counter": touchdown_counter.copy(),
                        "touchdown_latched": touchdown_latched.copy(),
                    }
                )

                if np.max(np.abs(rpy[:2])) > float(self.cfg.roll_pitch_abort_rad):
                    terminated = True
                    term_reason = "roll_pitch_abort"
                    break

                if touchdown_all and hold_counter >= int(self.cfg.post_touchdown_hold_steps):
                    break

            if terminated:
                break

        return {
            "qpos": np.asarray(qpos_hist),
            "qvel": np.asarray(qvel_hist),
            "base_rpy": np.asarray(rpy_hist),
            "contact_legs": contact_hist,
            "debug": debug_hist,
            "terminated": terminated,
            "termination_reason": term_reason,
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
            elif name.startswith("stepping_stone_") or name.startswith("bank_"):
                self.support_geom_ids.add(int(gid))
                self.world_geom_ids.add(int(gid))
            elif name == "river_bottom":
                self.world_geom_ids.add(int(gid))

    # ------------------------------------------------------------------
    # Plan decoding
    # ------------------------------------------------------------------
    def _decode_plan(self, plan_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, np.ndarray]]]:
        mids_xy, yaws, feet, _phase = decode_plan_states(plan_states, step_width=float(self.cfg.step_width))
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
                goal = self._project_to_safe_support(goal)
                goal[2] = max(float(self.cfg.min_foot_z), float(goal[2]))
                phase_goals[leg] = goal
            leg_goals.append(phase_goals)
        return mids_xy, yaws, leg_goals

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
    ) -> PhaseState:
        stance_legs = tuple(leg for leg in LEG_ORDER if leg not in swing_legs)
        swing_start = {leg: self._current_foot_pos(leg) for leg in swing_legs}
        swing_goal = {leg: np.asarray(leg_goals1[leg], dtype=np.float64).copy() for leg in swing_legs}
        stance_anchor = {leg: self._stance_anchor_target(self._current_foot_pos(leg)) for leg in stance_legs}

        for leg in stance_legs:
            stance_anchor[leg] = self._stance_anchor_target(stance_anchor[leg])
        for leg in swing_legs:
            swing_goal[leg] = self._project_to_safe_support(swing_goal[leg])
            swing_goal[leg][2] = max(float(self.cfg.min_foot_z), float(swing_goal[leg][2]))

        return PhaseState(
            swing_legs=swing_legs,
            stance_legs=stance_legs,
            start_mid_xy=np.asarray(mid0, dtype=np.float64).copy(),
            goal_mid_xy=np.asarray(mid1, dtype=np.float64).copy(),
            start_yaw=float(yaw0),
            goal_yaw=float(yaw1),
            swing_start=swing_start,
            swing_goal=swing_goal,
            stance_anchor=stance_anchor,
        )

    def _build_foot_targets(self, phase: PhaseState, alpha: float) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for leg in phase.stance_legs:
            out[leg] = phase.stance_anchor[leg].copy()
        for leg in phase.swing_legs:
            out[leg] = self._swing_target(
                phase.swing_start[leg],
                phase.swing_goal[leg],
                alpha,
                apex_z=max(
                    float(self.cfg.min_foot_z),
                    float(max(phase.swing_start[leg][2], phase.swing_goal[leg][2]) + self.cfg.swing_height),
                ),
            )
        return out

    def _build_foot_velocity_refs(self, phase: PhaseState, alpha: float) -> Dict[str, np.ndarray]:
        eps = 1.0 / max(float(self.cfg.phase_steps), 1.0)
        out: Dict[str, np.ndarray] = {}
        for leg in phase.stance_legs:
            out[leg] = np.zeros(3, dtype=np.float64)
        for leg in phase.swing_legs:
            a0 = np.clip(alpha - eps, 0.0, 1.0)
            a1 = np.clip(alpha + eps, 0.0, 1.0)
            z_apex = max(
                float(self.cfg.min_foot_z),
                float(max(phase.swing_start[leg][2], phase.swing_goal[leg][2]) + self.cfg.swing_height),
            )
            p0 = self._swing_target(
                phase.swing_start[leg],
                phase.swing_goal[leg],
                float(a0),
                apex_z=z_apex,
            )
            p1 = self._swing_target(
                phase.swing_start[leg],
                phase.swing_goal[leg],
                float(a1),
                apex_z=z_apex,
            )
            out[leg] = (p1 - p0) / max((a1 - a0) * float(self.cfg.phase_steps) * float(self.cfg.sim_dt), 1e-6)
        return out

    def _swing_target(
        self,
        start: np.ndarray,
        land_goal: np.ndarray,
        alpha: float,
        *,
        apex_z: float,
    ) -> np.ndarray:
        a = _smoothstep(alpha)
        finish_alpha = float(np.clip(self.cfg.swing_xy_finish_alpha, 0.05, 0.98))
        land_goal_arr = np.asarray(land_goal[:2], dtype=np.float64)
        if a <= finish_alpha:
            xy_alpha = a / max(finish_alpha, 1e-6)
            xy = (1.0 - xy_alpha) * np.asarray(start[:2], dtype=np.float64) + xy_alpha * land_goal_arr
        else:
            xy = land_goal_arr.copy()
        if a < 0.5:
            z = (1.0 - 2.0 * a) * float(start[2]) + (2.0 * a) * float(apex_z)
        else:
            tail = float(np.clip(self.cfg.swing_z_only_tail, 0.05, 0.45))
            descend_alpha = (a - 0.5) / 0.5
            descend_end = max(1.0 - tail, 0.55)
            if descend_alpha <= descend_end:
                u = descend_alpha / max(descend_end, 1e-6)
                z = (1.0 - u) * float(apex_z) + u * float(land_goal[2])
            else:
                z = float(land_goal[2])
        return np.array([xy[0], xy[1], max(float(self.cfg.min_foot_z), z)], dtype=np.float64)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def _compose_leg_torques(
        self,
        *,
        phase: PhaseState,
        target_feet: Dict[str, np.ndarray],
        foot_vel_ref: Dict[str, np.ndarray],
    ) -> np.ndarray:
        tau = np.zeros((len(self.actuator_ids),), dtype=np.float64)
        leg_to_slice = {
            "FL": slice(0, 3),
            "FR": slice(3, 6),
            "RL": slice(6, 9),
            "RR": slice(9, 12),
        }

        for leg in phase.stance_legs:
            tau_leg = self._task_space_leg_torque(
                leg,
                target_feet[leg],
                foot_vel_ref[leg],
                kp_xy=float(self.cfg.stance_kp_xy),
                kp_z=float(self.cfg.stance_kp_z),
                kd_xy=float(self.cfg.stance_kd_xy),
                kd_z=float(self.cfg.stance_kd_z),
                clip_xy=float(self.cfg.max_anchor_error_xy),
            )
            tau[leg_to_slice[leg]] += tau_leg

        for leg in phase.swing_legs:
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

    def _apply_base_pd(self, base_xy: np.ndarray, base_yaw: float, support_contacts: int) -> None:
        cfg = self.cfg
        q = self.data.qpos
        v = self.data.qvel
        rpy = _quat_to_rpy_wxyz(q[3:7])
        yaw_err = _wrap_angle(float(base_yaw) - float(rpy[2]))

        if support_contacts < int(cfg.base_support_contact_min):
            support_scale = 0.0
        else:
            support_scale = float(np.clip(float(support_contacts) / 4.0, 0.0, 1.0))

        fx = float(cfg.base_kp_xy) * (float(base_xy[0]) - float(q[0])) - float(cfg.base_kd_xy) * float(v[0])
        fy = float(cfg.base_kp_xy) * (float(base_xy[1]) - float(q[1])) - float(cfg.base_kd_xy) * float(v[1])
        z_ref = self._base_height + float(cfg.base_height_offset)
        gmag = float(abs(self.model.opt.gravity[2]))
        fz = (
            float(cfg.base_weight_comp) * self._total_mass * gmag
            + float(cfg.base_kp_z) * (z_ref - float(q[2]))
            - float(cfg.base_kd_z) * float(v[2])
        )
        tx = -float(cfg.base_kp_rp) * float(rpy[0]) - float(cfg.base_kd_rp) * float(v[3])
        ty = -float(cfg.base_kp_rp) * float(rpy[1]) - float(cfg.base_kd_rp) * float(v[4])
        tz = float(cfg.base_kp_yaw) * yaw_err - float(cfg.base_kd_yaw) * float(v[5])

        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[0] = support_scale * np.clip(fx, -cfg.base_force_xy_clip, cfg.base_force_xy_clip)
        self.data.qfrc_applied[1] = support_scale * np.clip(fy, -cfg.base_force_xy_clip, cfg.base_force_xy_clip)
        self.data.qfrc_applied[2] = support_scale * np.clip(fz, -cfg.base_force_z_clip, cfg.base_force_z_clip)
        self.data.qfrc_applied[3] = support_scale * np.clip(tx, -cfg.base_torque_clip, cfg.base_torque_clip)
        self.data.qfrc_applied[4] = support_scale * np.clip(ty, -cfg.base_torque_clip, cfg.base_torque_clip)
        self.data.qfrc_applied[5] = support_scale * np.clip(tz, -cfg.base_torque_clip, cfg.base_torque_clip)

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

    # ------------------------------------------------------------------
    # Contact sensing (mj_contactForce version)
    # ------------------------------------------------------------------
    def _clear_contact_cache(self) -> None:
        self._contact_wrench_by_leg = {leg: np.zeros(6, dtype=np.float64) for leg in LEG_ORDER}
        self._normal_force_by_leg = {leg: 0.0 for leg in LEG_ORDER}
        self._tangent_speed_by_leg = {leg: 0.0 for leg in LEG_ORDER}

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
            self._normal_force_by_leg[leg] += max(0.0, float(force_contact[0]))

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
        return [leg for leg in LEG_ORDER if self._has_contact_with_geom_set(leg, self.support_geom_ids)]

    def _touchdown_candidate_ready(self, leg: str, target_foot: np.ndarray) -> bool:
        in_contact = leg in self._ground_contact_legs()
        normal_ok = float(self._normal_force_by_leg[leg]) >= float(self.cfg.touchdown_force_thresh)
        tangent_ok = float(self._tangent_speed_by_leg[leg]) <= float(self.cfg.touchdown_tangent_speed_thresh)
        pos = self._current_foot_pos(leg)
        xy_ok = float(np.linalg.norm(pos[:2] - np.asarray(target_foot[:2], dtype=np.float64))) <= float(self.cfg.touchdown_xy_tol)
        return bool(in_contact and normal_ok and tangent_ok and xy_ok)

    def _strong_touchdown_ready(self, leg: str) -> bool:
        in_contact = leg in self._ground_contact_legs()
        normal_ok = float(self._normal_force_by_leg[leg]) >= float(self.cfg.strong_touchdown_force_thresh)
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
        out = self._project_to_safe_support(point)
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
            "base_rpy": np.asarray([_quat_to_rpy_wxyz(self.data.qpos[3:7])]),
            "contact_legs": [self._ground_contact_legs()],
            "debug": [],
            "terminated": False,
            "termination_reason": None,
        }