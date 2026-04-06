"""
MuJoCo pipeline for humanoid corridor followers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from genedynamics.deploy.followers.common.plan_adapter import CorridorTrajectoryAdapter
from genedynamics.deploy.followers.common.plan_schema import CorridorPlanSchema, CorridorPlanTrajectory
from genedynamics.deploy.followers.common.traversal_intent import TraversalIntent
from genedynamics.deploy.followers.humanoid.contact_scheduler import (
    HumanoidContactScheduler,
    HumanoidContactSchedulerConfig,
)
from genedynamics.deploy.followers.humanoid.footstep_planner import (
    FootstepPlannerConfig,
    HumanoidFootstepPlanner,
)
from genedynamics.deploy.followers.humanoid.models.g1_model import (
    G1ModelSpec,
    resolve_g1_model_path,
)
from genedynamics.deploy.followers.humanoid.mujoco.wbc_solver import (
    G1WBCTaskStackConfig,
    G1WholeBodySolverSkeleton,
)
from genedynamics.deploy.followers.humanoid.task_spec import (
    ContactObservations,
    FootContactObservation,
)
from genedynamics.deploy.followers.humanoid.task_builder import (
    HumanoidTaskBuilder,
    HumanoidTaskBuilderConfig,
)
from genedynamics.deploy.followers.humanoid.upper_body_mapper import (
    HumanoidUpperBodyMapper,
    HumanoidUpperBodyMapperConfig,
)


@dataclass
class HumanoidMujocoPipelineConfig:
    source_dt: float = 0.25
    control_dt: float = 0.02
    sim_dt: float = 0.002
    schema: CorridorPlanSchema = field(default_factory=CorridorPlanSchema.corridor_14d)
    contact_scheduler: HumanoidContactSchedulerConfig = field(default_factory=HumanoidContactSchedulerConfig)
    footstep_planner: FootstepPlannerConfig = field(default_factory=FootstepPlannerConfig)
    task_builder: HumanoidTaskBuilderConfig = field(default_factory=HumanoidTaskBuilderConfig)
    upper_body: HumanoidUpperBodyMapperConfig = field(default_factory=HumanoidUpperBodyMapperConfig)
    wbc: G1WBCTaskStackConfig = field(default_factory=G1WBCTaskStackConfig)
    startup_actuation_second_step_scale: float = 0.70


class HumanoidMujocoPipeline:
    """
    WBC-first MuJoCo pipeline for humanoid corridor following.

    Current execution order:
    plan adapter -> traversal intent -> contact scheduler -> footstep planner
    -> humanoid task builder -> inverse-dynamics WBC -> MuJoCo
    """

    def __init__(
        self,
        model_xml_path: Optional[str] = None,
        *,
        cfg: Optional[HumanoidMujocoPipelineConfig] = None,
        adapter: Optional[CorridorTrajectoryAdapter] = None,
        contact_scheduler: Optional[HumanoidContactScheduler] = None,
        footstep_planner: Optional[HumanoidFootstepPlanner] = None,
        task_builder: Optional[HumanoidTaskBuilder] = None,
        upper_body_mapper: Optional[HumanoidUpperBodyMapper] = None,
        solver: Optional[G1WholeBodySolverSkeleton] = None,
    ) -> None:
        import mujoco

        self.mujoco = mujoco
        self.cfg = cfg or HumanoidMujocoPipelineConfig()
        model_path = resolve_g1_model_path(model_xml_path)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = float(self.cfg.sim_dt)
        self.data = mujoco.MjData(self.model)
        self.model_spec = G1ModelSpec.from_mujoco_model(self.model, model_xml_path=model_path)

        self.adapter = adapter or CorridorTrajectoryAdapter(
            schema=self.cfg.schema,
            source_dt=float(self.cfg.source_dt),
            target_dt=float(self.cfg.control_dt),
        )
        self.contact_scheduler = contact_scheduler or HumanoidContactScheduler(self.cfg.contact_scheduler)
        self.footstep_planner = footstep_planner or HumanoidFootstepPlanner(self.cfg.footstep_planner)
        self.task_builder = task_builder or HumanoidTaskBuilder(self.cfg.task_builder)
        self.upper_body_mapper = upper_body_mapper or HumanoidUpperBodyMapper(self.cfg.upper_body)
        self.solver = solver or G1WholeBodySolverSkeleton(
            self.model,
            self.data,
            self.model_spec,
            self.cfg.wbc,
        )

        self._stand_qpos = np.zeros(self.model.nq, dtype=np.float64)
        self._render_root_height_offset = 0.0

        self.reset()

    def reset(self, initial_frame: Optional[Any] = None) -> None:
        key_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_KEY, "stand")
        if key_id >= 0:
            self.mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            self.mujoco.mj_resetData(self.model, self.data)
        if initial_frame is not None:
            self._align_root_to_plan_frame(initial_frame)
        self.mujoco.mj_forward(self.model, self.data)
        self._stand_qpos = self.data.qpos.copy()
        self._render_root_height_offset = float(self._stand_qpos[2]) - float(self.cfg.upper_body.body_height_nominal)
        left_foot, right_foot = self.current_foot_world_positions()
        initial_intent = None if initial_frame is None else TraversalIntent.from_plan_frame(initial_frame)
        self.contact_scheduler.reset(initial_frame=initial_frame)
        self.footstep_planner.reset(initial_intent=initial_intent)
        self.footstep_planner.seed_from_current_feet(left_foot, right_foot)

    def load_plan_from_seed_dir(
        self,
        seed_dir: str,
        *,
        best_idx: Optional[int] = None,
    ) -> CorridorPlanTrajectory:
        return self.adapter.load_seed_dir(seed_dir, best_idx=best_idx)

    def rollout_plan_from_seed_dir(
        self,
        seed_dir: str,
        *,
        best_idx: Optional[int] = None,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        traj = self.load_plan_from_seed_dir(seed_dir, best_idx=best_idx)
        return self.rollout_plan(traj, max_frames=max_frames)

    def rollout_plan(
        self,
        traj: CorridorPlanTrajectory,
        *,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        frames = traj.decode_frames()
        if max_frames is not None:
            frames = frames[: int(max_frames)]
        if not frames:
            return self._empty_rollout()
        self.reset(initial_frame=frames[0])

        qpos_hist = []
        qvel_hist = []
        ctrl_hist = []
        tau_hist = []
        lambda_hist = []
        ddq_hist = []
        state_hist = []
        render_state_hist = []
        plan_states = []
        debug = []

        step_dt = float(max(traj.dt, self.cfg.control_dt))
        sim_steps = max(1, int(round(step_dt / float(self.model.opt.timestep))))

        for i, frame in enumerate(frames):
            intent = TraversalIntent.from_plan_frame(frame)
            contact_obs = self.current_contact_observations()
            phase = self.contact_scheduler.advance(intent, step_dt, contact_obs)
            footsteps = self.footstep_planner.update(intent, phase, contact_obs, dt=step_dt)
            upper_body = self.upper_body_mapper.map(frame)
            tasks = self.task_builder.build(frame, intent, phase, footsteps, upper_body)
            targets = self.solver.solve(tasks, qpos=self.data.qpos, qvel=self.data.qvel, dt=step_dt)
            self._apply_startup_actuation_blend(targets, phase)

            render_state_hist.append(self._compose_render_state(targets.q_ref, tasks))
            self.data.qfrc_applied[:] = 0.0
            self.data.ctrl[:] = 0.0
            self.data.ctrl[: targets.q_ref.size] = targets.q_ref
            if targets.tau_ff is not None:
                self.data.qfrc_applied[self.model_spec.actuated_dof_indices] = np.asarray(targets.tau_ff, dtype=np.float64)
            for _ in range(sim_steps):
                self.mujoco.mj_step(self.model, self.data)

            qpos_hist.append(self.data.qpos.copy())
            qvel_hist.append(self.data.qvel.copy())
            ctrl_hist.append(targets.q_ref.copy())
            tau_hist.append(
                np.zeros((self.model_spec.num_actuated,), dtype=np.float64)
                if targets.tau_ff is None
                else np.asarray(targets.tau_ff, dtype=np.float64).copy()
            )
            ddq_hist.append(
                np.zeros((self.model.nv,), dtype=np.float64)
                if targets.ddq_ref is None
                else np.asarray(targets.ddq_ref, dtype=np.float64).copy()
            )
            lambda_hist.append(self._pad_lambda(targets))
            state_hist.append(np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()], axis=0))
            plan_states.append(frame.raw_state.copy())
            debug.append(
                {
                    "frame_index": i,
                    "time_sec": float(frame.time_sec),
                    "phase": phase.phase.value,
                    "phase_alpha": float(phase.alpha),
                    "intent": {
                        "planar_position": intent.planar_position.copy(),
                        "yaw": float(intent.yaw),
                        "body_height": float(intent.body_height),
                        "planar_velocity": intent.planar_velocity.copy(),
                        "yaw_rate": float(intent.yaw_rate),
                    },
                    "footsteps": {
                        "left_position_world": footsteps.left_position_world.copy(),
                        "right_position_world": footsteps.right_position_world.copy(),
                        "active_swing_foot": footsteps.active_swing_foot,
                        "swing_goal_world": None if footsteps.swing_goal_world is None else footsteps.swing_goal_world.copy(),
                    },
                    "pelvis_target": tasks.pelvis.position_world.copy(),
                    "torso_yaw_target": float(tasks.torso_yaw),
                    "solver_metadata": dict(targets.metadata),
                }
            )

        return {
            "states": np.asarray(state_hist, dtype=np.float64),
            "qpos": np.asarray(qpos_hist, dtype=np.float64),
            "qvel": np.asarray(qvel_hist, dtype=np.float64),
            "ctrl": np.asarray(ctrl_hist, dtype=np.float64),
            "tau_ff": np.asarray(tau_hist, dtype=np.float64),
            "lambda": np.asarray(lambda_hist, dtype=np.float64),
            "ddq": np.asarray(ddq_hist, dtype=np.float64),
            "render_states": np.asarray(render_state_hist, dtype=np.float64),
            "plan_states": np.asarray(plan_states, dtype=np.float64),
            "debug": debug,
            "metadata": {
                "model_xml_path": self.model_spec.model_xml_path,
                "control_dt": step_dt,
                "sim_dt": float(self.model.opt.timestep),
                "num_frames": len(frames),
                "best_idx": int(traj.best_idx),
                "schema_version": self.cfg.schema.version,
                "render_state_mode": "wbc_inverse_dynamics_root_preview",
                "solver": getattr(self.solver, "solver_name", "unknown_solver"),
            },
        }

    def current_foot_world_positions(self) -> tuple[np.ndarray, np.ndarray]:
        left_sid = self.model_spec.site_id.get("left_foot")
        right_sid = self.model_spec.site_id.get("right_foot")
        if left_sid is None or right_sid is None:
            raise KeyError("Left/right foot sites are required for the humanoid MuJoCo pipeline")
        left = np.asarray(self.data.site_xpos[left_sid], dtype=np.float64).copy()
        right = np.asarray(self.data.site_xpos[right_sid], dtype=np.float64).copy()
        return left, right

    def _apply_startup_actuation_blend(self, targets: Any, phase: Any) -> None:
        beta = self._startup_actuation_blend(phase)
        if beta >= 0.999:
            return
        stand = np.asarray(self.model_spec.stand_ctrl, dtype=np.float64)
        upper_mask = np.zeros((self.model_spec.num_actuated,), dtype=np.float64)
        upper_joint_names = (
            tuple(self.model_spec.waist_joints)
            + tuple(self.model_spec.left_arm_joints)
            + tuple(self.model_spec.right_arm_joints)
        )
        for name in upper_joint_names:
            if name in self.model_spec.actuated_joints:
                upper_mask[self.model_spec.actuated_joints.index(name)] = 1.0
        leg_mask = 1.0 - upper_mask
        leg_beta = min(1.0, 0.70 + 0.30 * beta)

        q_ref = np.asarray(targets.q_ref, dtype=np.float64)
        blended_q = q_ref.copy()
        blended_q = stand + upper_mask * beta * (blended_q - stand) + leg_mask * leg_beta * (blended_q - stand)
        targets.q_ref = blended_q
        if targets.qd_ref is not None:
            qd_ref = np.asarray(targets.qd_ref, dtype=np.float64)
            targets.qd_ref = upper_mask * beta * qd_ref + leg_mask * leg_beta * qd_ref
        if targets.ddq_ref is not None:
            ddq_ref = np.asarray(targets.ddq_ref, dtype=np.float64)
            ddq_ref = ddq_ref.copy()
            act_idx = np.asarray(self.model_spec.actuated_dof_indices, dtype=np.int32)
            ddq_ref[act_idx] = upper_mask * beta * ddq_ref[act_idx] + leg_mask * leg_beta * ddq_ref[act_idx]
            targets.ddq_ref = ddq_ref
        if targets.tau_ff is not None:
            tau_ff = np.asarray(targets.tau_ff, dtype=np.float64)
            targets.tau_ff = upper_mask * beta * tau_ff + leg_mask * leg_beta * tau_ff
        targets.metadata = dict(targets.metadata)
        targets.metadata["startup_actuation_blend"] = float(beta)
        targets.metadata["startup_leg_actuation_blend"] = float(leg_beta)

    def _startup_actuation_blend(self, phase: Any) -> float:
        if int(phase.step_index) <= 0:
            if phase.phase == "double_support" or getattr(phase.phase, "value", None) == "double_support":
                alpha = float(np.clip(phase.alpha, 0.0, 1.0))
                return float(0.10 + 0.50 * alpha)
            liftoff = bool(phase.metadata.get("liftoff_confirmed", False))
            if not liftoff:
                return 0.35
            release = float(np.clip((phase.alpha - 0.18) / 0.82, 0.0, 1.0))
            return float(0.80 + 0.20 * release)
        if int(phase.step_index) == 1:
            alpha = float(np.clip(phase.alpha, 0.0, 1.0))
            return float(min(1.0, self.cfg.startup_actuation_second_step_scale + 0.45 * alpha))
        return 1.0

    def current_contact_observations(self) -> ContactObservations:
        left = self._foot_contact_observation("left")
        right = self._foot_contact_observation("right")
        return ContactObservations(left=left, right=right)

    @staticmethod
    def _empty_rollout() -> Dict[str, Any]:
        return {
            "states": np.zeros((0, 0), dtype=np.float64),
            "qpos": np.zeros((0, 0), dtype=np.float64),
            "qvel": np.zeros((0, 0), dtype=np.float64),
            "ctrl": np.zeros((0, 0), dtype=np.float64),
            "tau_ff": np.zeros((0, 0), dtype=np.float64),
            "lambda": np.zeros((0, 0), dtype=np.float64),
            "ddq": np.zeros((0, 0), dtype=np.float64),
            "render_states": np.zeros((0, 0), dtype=np.float64),
            "plan_states": np.zeros((0, 0), dtype=np.float64),
            "debug": [],
            "metadata": {"num_frames": 0},
        }

    def _align_root_to_plan_frame(self, frame: Any) -> None:
        if self.data.qpos.size < 7:
            return
        self.data.qpos[0] = float(frame.x)
        self.data.qpos[1] = float(frame.y)
        self.data.qpos[2] = float(frame.h) + self._infer_root_height_offset()
        self.data.qpos[3:7] = self._quat_wxyz_from_rpy(0.0, 0.0, float(frame.psi))
        if self.model_spec.stand_ctrl.size == self.model_spec.num_actuated:
            self.data.qpos[self.model_spec.actuated_qpos_indices] = self.model_spec.stand_ctrl.copy()

    def _foot_contact_observation(self, side: str) -> FootContactObservation:
        site_name = f"{side}_foot"
        body_name = f"{side}_ankle_roll_link"
        sid = self.model_spec.site_id.get(site_name)
        bid = self.model_spec.body_id.get(body_name)
        if sid is None:
            raise KeyError(f"Missing foot site: {site_name}")
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        self.mujoco.mj_jacSite(self.model, self.data, jacp, jacr, sid)
        vel = jacp @ self.data.qvel
        ang_vel = jacr @ self.data.qvel
        in_contact = False
        contact_count = 0
        support_load = 0.0
        if bid is not None:
            for ci in range(int(self.data.ncon)):
                contact = self.data.contact[ci]
                body1 = int(self.model.geom_bodyid[int(contact.geom1)])
                body2 = int(self.model.geom_bodyid[int(contact.geom2)])
                if bid not in (body1, body2):
                    continue
                in_contact = True
                contact_count += 1
                wrench = np.zeros((6,), dtype=np.float64)
                self.mujoco.mj_contactForce(self.model, self.data, ci, wrench)
                support_load += max(float(wrench[0]), 0.0)
        return FootContactObservation(
            position_world=np.asarray(self.data.site_xpos[sid], dtype=np.float64).copy(),
            velocity_world=np.asarray(vel, dtype=np.float64).copy(),
            rotation_world=np.asarray(self.data.site_xmat[sid], dtype=np.float64).reshape(3, 3).copy(),
            angular_velocity_world=np.asarray(ang_vel, dtype=np.float64).copy(),
            in_contact=in_contact,
            contact_count=contact_count,
            support_load=support_load,
        )

    @staticmethod
    def _pad_lambda(targets: Any) -> np.ndarray:
        lam = np.zeros((6,), dtype=np.float64)
        if getattr(targets, "lambda_ref", None) is None:
            return lam
        raw = np.asarray(targets.lambda_ref, dtype=np.float64).reshape(-1)
        meta = getattr(targets, "metadata", {}) or {}
        contact_feet = list(meta.get("contact_feet", []))
        for i, name in enumerate(contact_feet):
            src = raw[3 * i: 3 * (i + 1)]
            if src.size != 3:
                continue
            dst = slice(0, 3) if name == "left_foot" else slice(3, 6)
            lam[dst] = src
        return lam

    def _infer_root_height_offset(self) -> float:
        key_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_KEY, "stand")
        if key_id >= 0 and self.model.key_qpos.shape[1] >= 3:
            return float(self.model.key_qpos[key_id, 2]) - float(self.cfg.upper_body.body_height_nominal)
        return float(self.data.qpos[2]) - float(self.cfg.upper_body.body_height_nominal)

    def _compose_render_state(self, q_ref: np.ndarray, tasks: Any) -> np.ndarray:
        qpos = self._stand_qpos.copy()
        pelvis = np.asarray(tasks.pelvis.position_world, dtype=np.float64).reshape(3)
        qpos[0] = float(pelvis[0])
        qpos[1] = float(pelvis[1])
        qpos[2] = float(pelvis[2]) + float(self._render_root_height_offset)
        qpos[3:7] = self._quat_wxyz_from_rpy(
            float(tasks.pelvis.roll_world),
            float(tasks.pelvis.pitch_world),
            float(tasks.pelvis.yaw_world),
        )
        qpos = self.model_spec.apply_actuated_qpos(qpos, q_ref)
        qvel = np.zeros(self.model.nv, dtype=np.float64)
        return np.concatenate([qpos, qvel], axis=0)

    @staticmethod
    def _quat_wxyz_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr = np.cos(0.5 * roll)
        sr = np.sin(0.5 * roll)
        cp = np.cos(0.5 * pitch)
        sp = np.sin(0.5 * pitch)
        cy = np.cos(0.5 * yaw)
        sy = np.sin(0.5 * yaw)
        return np.asarray(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=np.float64,
        )
