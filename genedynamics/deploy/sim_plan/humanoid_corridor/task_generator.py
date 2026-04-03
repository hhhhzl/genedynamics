"""
Task-space generation from typed corridor plan frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from genedynamics.deploy.sim_plan.humanoid_corridor.schema import (
    ArmJointTask,
    ContactPhase,
    CorridorPlanFrame,
    FollowerTasks,
    FootTask,
    PelvisTask,
    PhaseState,
)


@dataclass
class CorridorTaskGeneratorConfig:
    nominal_stance_width: float = 0.24
    nominal_foot_x: float = 0.02
    swing_height: float = 0.05
    footstep_preview_time: float = 0.35
    max_step_forward: float = 0.40
    max_step_backward: float = 0.10
    max_step_lateral: float = 0.14
    max_pelvis_forward_offset: float = 0.20
    max_pelvis_backward_offset: float = 0.08
    max_pelvis_lateral_offset: float = 0.12
    torso_yaw_gain: float = 1.0
    body_pitch_from_crouch_gain: float = 0.08
    body_height_nominal: float = 0.75
    body_height_min: float = 0.55
    arm_tuck_elbow_gain: float = 1.10
    arm_tuck_shoulder_roll_gain: float = 0.55
    arm_posture_pitch_gain: float = 0.45
    arm_posture_yaw_gain: float = 0.35


class CorridorTaskGenerator:
    """
    Convert corridor plan frames into solver-facing tasks.

    The generator owns foot anchor state so that the phase scheduler can stay
    simple and purely temporal in stage 1.
    """

    def __init__(self, cfg: Optional[CorridorTaskGeneratorConfig] = None) -> None:
        self.cfg = cfg or CorridorTaskGeneratorConfig()
        self._left_anchor_world: Optional[np.ndarray] = None
        self._right_anchor_world: Optional[np.ndarray] = None
        self._left_nominal_z: float = 0.0
        self._right_nominal_z: float = 0.0
        self._swing_start_world: Optional[np.ndarray] = None
        self._swing_goal_world: Optional[np.ndarray] = None
        self._active_swing_foot: Optional[str] = None
        self._last_phase: Optional[ContactPhase] = None

    def reset(
        self,
        initial_frame: Optional[CorridorPlanFrame] = None,
        *,
        left_anchor_world: Optional[np.ndarray] = None,
        right_anchor_world: Optional[np.ndarray] = None,
    ) -> None:
        if left_anchor_world is not None and right_anchor_world is not None:
            self._left_anchor_world = np.asarray(left_anchor_world, dtype=np.float64).copy()
            self._right_anchor_world = np.asarray(right_anchor_world, dtype=np.float64).copy()
            self._left_nominal_z = float(self._left_anchor_world[2])
            self._right_nominal_z = float(self._right_anchor_world[2])
        elif initial_frame is not None:
            self._left_anchor_world, self._right_anchor_world = self._default_anchor_positions(initial_frame)
        else:
            self._left_anchor_world = None
            self._right_anchor_world = None
            self._left_nominal_z = 0.0
            self._right_nominal_z = 0.0
        self._swing_start_world = None
        self._swing_goal_world = None
        self._active_swing_foot = None
        self._last_phase = None

    def seed_from_current_feet(self, left_world: np.ndarray, right_world: np.ndarray) -> None:
        self.reset(
            left_anchor_world=np.asarray(left_world, dtype=np.float64),
            right_anchor_world=np.asarray(right_world, dtype=np.float64),
        )

    def generate(
        self,
        frame: CorridorPlanFrame,
        phase: PhaseState,
        *,
        dt: float,
        previous_tasks: Optional[FollowerTasks] = None,
    ) -> FollowerTasks:
        if self._left_anchor_world is None or self._right_anchor_world is None:
            self._left_anchor_world, self._right_anchor_world = self._default_anchor_positions(frame)

        transitioned = phase.metadata.get("transitioned", False)
        if transitioned:
            self._handle_transition(frame, phase)

        left_pos, left_vel = self._foot_task_for_side("left", frame, phase, dt)
        right_pos, right_vel = self._foot_task_for_side("right", frame, phase, dt)
        pelvis_xy = self._clamp_pelvis_xy_to_support(frame, left_pos, right_pos)

        crouch_ratio = self._crouch_ratio(frame.h)
        pelvis = PelvisTask(
            position_world=np.asarray([pelvis_xy[0], pelvis_xy[1], frame.h], dtype=np.float64),
            yaw_world=float(frame.psi),
            roll_world=0.0,
            pitch_world=float(self.cfg.body_pitch_from_crouch_gain) * crouch_ratio,
        )
        left_arm = ArmJointTask(joint_targets=self._arm_joint_targets("left", frame.a_left, frame.p_left))
        right_arm = ArmJointTask(joint_targets=self._arm_joint_targets("right", frame.a_right, frame.p_right))

        joint_hints: Dict[str, float] = {
            "waist_yaw_joint": float(self.cfg.torso_yaw_gain) * float(frame.psi_torso),
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": float(self.cfg.body_pitch_from_crouch_gain) * crouch_ratio,
        }
        joint_hints.update(left_arm.joint_targets)
        joint_hints.update(right_arm.joint_targets)

        tasks = FollowerTasks(
            plan_frame=frame,
            pelvis=pelvis,
            torso_yaw=joint_hints["waist_yaw_joint"],
            left_foot=FootTask(
                position_world=left_pos,
                velocity_world=left_vel,
                yaw_world=float(frame.psi),
                in_contact=phase.phase != ContactPhase.LEFT_SWING,
                weight=1.0 if phase.phase != ContactPhase.LEFT_SWING else 0.35,
            ),
            right_foot=FootTask(
                position_world=right_pos,
                velocity_world=right_vel,
                yaw_world=float(frame.psi),
                in_contact=phase.phase != ContactPhase.RIGHT_SWING,
                weight=1.0 if phase.phase != ContactPhase.RIGHT_SWING else 0.35,
            ),
            left_arm=left_arm,
            right_arm=right_arm,
            joint_hints=joint_hints,
            extras={
                "crouch_ratio": crouch_ratio,
                "phase": phase.phase.value,
                "step_index": phase.step_index,
                "previous_tasks": previous_tasks is not None,
            },
        )
        self._last_phase = phase.phase
        return tasks

    def _handle_transition(self, frame: CorridorPlanFrame, phase: PhaseState) -> None:
        if phase.phase == ContactPhase.DOUBLE_SUPPORT and self._active_swing_foot and self._swing_goal_world is not None:
            if self._active_swing_foot == "left":
                self._left_anchor_world = self._swing_goal_world.copy()
            else:
                self._right_anchor_world = self._swing_goal_world.copy()
            self._active_swing_foot = None
            self._swing_start_world = None
            self._swing_goal_world = None
            return

        if phase.phase == ContactPhase.LEFT_SWING:
            self._active_swing_foot = "left"
            self._swing_start_world = self._left_anchor_world.copy()
            self._swing_goal_world = self._proposed_step_world(frame, "left")
        elif phase.phase == ContactPhase.RIGHT_SWING:
            self._active_swing_foot = "right"
            self._swing_start_world = self._right_anchor_world.copy()
            self._swing_goal_world = self._proposed_step_world(frame, "right")

    def _foot_task_for_side(
        self,
        side: str,
        frame: CorridorPlanFrame,
        phase: PhaseState,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        _ = dt
        anchor = self._left_anchor_world if side == "left" else self._right_anchor_world
        if anchor is None:
            anchor = self._default_anchor_positions(frame)[0 if side == "left" else 1]
        if phase.swing_foot != side or self._swing_start_world is None or self._swing_goal_world is None:
            return np.asarray(anchor, dtype=np.float64).copy(), np.zeros(3, dtype=np.float64)

        alpha = float(np.clip(phase.alpha, 0.0, 1.0))
        blend = alpha * alpha * (3.0 - 2.0 * alpha)
        pos = (1.0 - blend) * self._swing_start_world + blend * self._swing_goal_world
        pos = np.asarray(pos, dtype=np.float64)
        pos[2] += float(self.cfg.swing_height) * np.sin(np.pi * alpha)
        vel = (self._swing_goal_world - self._swing_start_world) / max(phase.phase_duration, 1e-6)
        vel = np.asarray(vel, dtype=np.float64)
        vel[2] = 0.0
        return pos, vel

    def _default_anchor_positions(self, frame: CorridorPlanFrame) -> Tuple[np.ndarray, np.ndarray]:
        yaw = float(frame.psi)
        rot = np.asarray(
            [
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)],
            ],
            dtype=np.float64,
        )
        left_local = np.asarray([self.cfg.nominal_foot_x, 0.5 * self.cfg.nominal_stance_width], dtype=np.float64)
        right_local = np.asarray([self.cfg.nominal_foot_x, -0.5 * self.cfg.nominal_stance_width], dtype=np.float64)
        left_xy = np.asarray([frame.x, frame.y], dtype=np.float64) + rot @ left_local
        right_xy = np.asarray([frame.x, frame.y], dtype=np.float64) + rot @ right_local
        left = np.asarray([left_xy[0], left_xy[1], 0.0], dtype=np.float64)
        right = np.asarray([right_xy[0], right_xy[1], 0.0], dtype=np.float64)
        left[2] = float(self._left_nominal_z)
        right[2] = float(self._right_nominal_z)
        return left, right

    def _proposed_step_world(self, frame: CorridorPlanFrame, side: str) -> np.ndarray:
        yaw = float(frame.psi)
        rot = np.asarray(
            [
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)],
            ],
            dtype=np.float64,
        )
        preview = float(self.cfg.footstep_preview_time)
        preview_xy = np.asarray([frame.v_x, frame.v_y], dtype=np.float64) * preview
        nominal_local = np.asarray(
            [
                self.cfg.nominal_foot_x + frame.v_x * preview,
                0.5 * self.cfg.nominal_stance_width if side == "left" else -0.5 * self.cfg.nominal_stance_width,
            ],
            dtype=np.float64,
        )
        nominal_xy = np.asarray([frame.x, frame.y], dtype=np.float64) + rot @ nominal_local + 0.2 * preview_xy
        anchor = self._left_anchor_world if side == "left" else self._right_anchor_world
        if anchor is None:
            return np.asarray([nominal_xy[0], nominal_xy[1], 0.0], dtype=np.float64)

        delta_xy = nominal_xy - np.asarray(anchor[:2], dtype=np.float64)
        local_delta = rot.T @ delta_xy
        local_delta[0] = float(np.clip(local_delta[0], -self.cfg.max_step_backward, self.cfg.max_step_forward))
        local_delta[1] = float(np.clip(local_delta[1], -self.cfg.max_step_lateral, self.cfg.max_step_lateral))
        clamped_xy = np.asarray(anchor[:2], dtype=np.float64) + rot @ local_delta
        nominal_z = float(self._left_nominal_z if side == "left" else self._right_nominal_z)
        return np.asarray([clamped_xy[0], clamped_xy[1], nominal_z], dtype=np.float64)

    def _clamp_pelvis_xy_to_support(
        self,
        frame: CorridorPlanFrame,
        left_pos: np.ndarray,
        right_pos: np.ndarray,
    ) -> np.ndarray:
        yaw = float(frame.psi)
        rot = np.asarray(
            [
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)],
            ],
            dtype=np.float64,
        )
        support_mid = 0.5 * (np.asarray(left_pos[:2], dtype=np.float64) + np.asarray(right_pos[:2], dtype=np.float64))
        desired = np.asarray([frame.x, frame.y], dtype=np.float64)
        local_offset = rot.T @ (desired - support_mid)
        local_offset[0] = float(
            np.clip(
                local_offset[0],
                -self.cfg.max_pelvis_backward_offset,
                self.cfg.max_pelvis_forward_offset,
            )
        )
        local_offset[1] = float(
            np.clip(
                local_offset[1],
                -self.cfg.max_pelvis_lateral_offset,
                self.cfg.max_pelvis_lateral_offset,
            )
        )
        return support_mid + rot @ local_offset

    def _arm_joint_targets(self, side: str, tuck: float, posture: float) -> Dict[str, float]:
        a = float(np.clip(tuck, 0.0, 1.0))
        p = float(np.clip(posture, -1.0, 1.0))
        roll_sign = 1.0 if side == "left" else -1.0
        prefix = "left" if side == "left" else "right"
        return {
            f"{prefix}_shoulder_pitch_joint": float(self.cfg.arm_posture_pitch_gain) * p,
            f"{prefix}_shoulder_roll_joint": roll_sign * float(self.cfg.arm_tuck_shoulder_roll_gain) * (1.0 - a),
            f"{prefix}_shoulder_yaw_joint": roll_sign * float(self.cfg.arm_posture_yaw_gain) * p,
            f"{prefix}_elbow_joint": float(self.cfg.arm_tuck_elbow_gain) * a,
            f"{prefix}_wrist_roll_joint": 0.0,
            f"{prefix}_wrist_pitch_joint": 0.0,
            f"{prefix}_wrist_yaw_joint": 0.0,
        }

    def _crouch_ratio(self, height: float) -> float:
        denom = max(self.cfg.body_height_nominal - self.cfg.body_height_min, 1e-6)
        return float(np.clip((self.cfg.body_height_nominal - float(height)) / denom, 0.0, 1.0))
