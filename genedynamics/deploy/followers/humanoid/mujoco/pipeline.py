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


class HumanoidMujocoPipeline:
    """
    WBC-first MuJoCo pipeline for humanoid corridor following.

    Current execution order:
    plan adapter -> traversal intent -> contact scheduler -> footstep planner
    -> humanoid task builder -> contact-aware WBC QP -> MuJoCo
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
        state_hist = []
        render_state_hist = []
        plan_states = []
        debug = []

        step_dt = float(max(traj.dt, self.cfg.control_dt))
        sim_steps = max(1, int(round(step_dt / float(self.model.opt.timestep))))

        for i, frame in enumerate(frames):
            intent = TraversalIntent.from_plan_frame(frame)
            phase = self.contact_scheduler.advance(frame, step_dt)
            footsteps = self.footstep_planner.update(intent, phase, dt=step_dt)
            upper_body = self.upper_body_mapper.map(frame)
            tasks = self.task_builder.build(frame, intent, phase, footsteps, upper_body)
            targets = self.solver.solve(tasks, qpos=self.data.qpos, qvel=self.data.qvel, dt=step_dt)

            render_state_hist.append(self._compose_render_state(targets.q_ref, tasks))
            self.data.ctrl[:] = 0.0
            self.data.ctrl[: targets.q_ref.size] = targets.q_ref
            for _ in range(sim_steps):
                self.mujoco.mj_step(self.model, self.data)

            qpos_hist.append(self.data.qpos.copy())
            qvel_hist.append(self.data.qvel.copy())
            ctrl_hist.append(targets.q_ref.copy())
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
                "render_state_mode": "wbc_root_preview",
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

    @staticmethod
    def _empty_rollout() -> Dict[str, Any]:
        return {
            "states": np.zeros((0, 0), dtype=np.float64),
            "qpos": np.zeros((0, 0), dtype=np.float64),
            "qvel": np.zeros((0, 0), dtype=np.float64),
            "ctrl": np.zeros((0, 0), dtype=np.float64),
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
