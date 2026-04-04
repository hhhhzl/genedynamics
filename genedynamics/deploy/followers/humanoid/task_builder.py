"""
Build humanoid task specs from footsteps, contact, and upper-body intent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from genedynamics.deploy.followers.common.plan_schema import CorridorPlanFrame
from genedynamics.deploy.followers.common.traversal_intent import TraversalIntent
from genedynamics.deploy.followers.humanoid.corridor_envelope import (
    CorridorEnvelopeConfig,
    CorridorEnvelopeHeuristics,
)
from genedynamics.deploy.followers.humanoid.footstep_planner import FootstepPlan
from genedynamics.deploy.followers.humanoid.task_spec import (
    ContactPhase,
    FootTask,
    HumanoidTaskSpec,
    PelvisTask,
    PhaseState,
)
from genedynamics.deploy.followers.humanoid.upper_body_mapper import UpperBodyTargets


@dataclass
class HumanoidTaskBuilderConfig:
    max_pelvis_forward_offset: float = 0.20
    max_pelvis_backward_offset: float = 0.08
    max_pelvis_lateral_offset: float = 0.12
    single_support_forward_offset: float = 0.05
    single_support_backward_offset: float = 0.02
    single_support_lateral_offset: float = 0.03
    pre_liftoff_forward_offset: float = 0.03
    pre_liftoff_backward_offset: float = 0.02
    pre_liftoff_lateral_offset: float = 0.025
    double_support_velocity_scale: float = 0.90
    single_support_velocity_scale: float = 0.30
    through_gap_single_support_velocity_scale: float = 0.18
    body_pitch_from_crouch_gain: float = 0.08
    pelvis_roll_from_lateral_velocity_gain: float = 0.08
    pelvis_roll_max: float = 0.18
    pelvis_pitch_max: float = 0.22
    through_gap_pelvis_lateral_scale: float = 0.45
    swing_task_min_weight: float = 0.04
    swing_task_max_weight: float = 0.28
    swing_liftoff_transition_alpha: float = 0.40
    envelope: CorridorEnvelopeConfig = field(default_factory=CorridorEnvelopeConfig)


class HumanoidTaskBuilder:
    def __init__(self, cfg: HumanoidTaskBuilderConfig | None = None) -> None:
        self.cfg = cfg or HumanoidTaskBuilderConfig()
        self.envelope = CorridorEnvelopeHeuristics(self.cfg.envelope)

    def build(
        self,
        frame: CorridorPlanFrame,
        intent: TraversalIntent,
        phase: PhaseState,
        footsteps: FootstepPlan,
        upper_body: UpperBodyTargets,
    ) -> HumanoidTaskSpec:
        envelope_state = self.envelope.evaluate(intent)
        left_in_contact, right_in_contact = self._contact_flags(phase)
        pelvis_xy = self._clamp_pelvis_xy_to_support(
            intent,
            phase,
            footsteps,
            left_in_contact,
            right_in_contact,
        )
        left_pos, left_vel, left_weight = self._foot_command("left", phase, footsteps, left_in_contact)
        right_pos, right_vel, right_weight = self._foot_command("right", phase, footsteps, right_in_contact)
        pelvis_velocity = self._pelvis_velocity_command(frame, intent, phase, envelope_state.gap_severity)
        pelvis = PelvisTask(
            position_world=np.asarray([pelvis_xy[0], pelvis_xy[1], intent.body_height], dtype=np.float64),
            yaw_world=float(intent.yaw),
            roll_world=float(
                np.clip(
                    -self.cfg.pelvis_roll_from_lateral_velocity_gain * float(intent.planar_velocity[1]),
                    -self.cfg.pelvis_roll_max,
                    self.cfg.pelvis_roll_max,
                )
            ),
            pitch_world=float(
                np.clip(
                    self.cfg.body_pitch_from_crouch_gain * float(upper_body.crouch_ratio),
                    -self.cfg.pelvis_pitch_max,
                    self.cfg.pelvis_pitch_max,
                )
            ),
            linear_velocity_world=pelvis_velocity,
            angular_velocity_world=np.asarray([0.0, 0.0, intent.yaw_rate], dtype=np.float64),
        )
        return HumanoidTaskSpec(
            plan_frame=frame,
            pelvis=pelvis,
            torso_yaw=float(upper_body.torso_yaw),
            left_foot=FootTask(
                position_world=left_pos,
                velocity_world=left_vel,
                yaw_world=float(intent.yaw),
                in_contact=left_in_contact,
                roll_world=0.0,
                pitch_world=0.0,
                angular_velocity_world=np.asarray([0.0, 0.0, intent.yaw_rate], dtype=np.float64),
                weight=left_weight,
            ),
            right_foot=FootTask(
                position_world=right_pos,
                velocity_world=right_vel,
                yaw_world=float(intent.yaw),
                in_contact=right_in_contact,
                roll_world=0.0,
                pitch_world=0.0,
                angular_velocity_world=np.asarray([0.0, 0.0, intent.yaw_rate], dtype=np.float64),
                weight=right_weight,
            ),
            left_arm=upper_body.left_arm,
            right_arm=upper_body.right_arm,
            joint_hints=dict(upper_body.joint_hints),
            extras={
                "crouch_ratio": float(upper_body.crouch_ratio),
                "narrowness": float(upper_body.narrowness),
                "gap_severity": float(upper_body.gap_severity),
                "effective_width": float(upper_body.effective_width),
                "through_gap_mode": bool(upper_body.through_gap_mode),
                "phase": phase.phase.value,
                "phase_alpha": float(phase.alpha),
                "step_index": phase.step_index,
                "left_in_contact": bool(left_in_contact),
                "right_in_contact": bool(right_in_contact),
                "footsteps": dict(footsteps.metadata),
                "envelope": {
                    "narrowness": float(envelope_state.narrowness),
                    "gap_severity": float(envelope_state.gap_severity),
                    "effective_width": float(envelope_state.effective_width),
                    "through_gap_mode": bool(envelope_state.through_gap_mode),
                },
            },
        )

    def _swing_task_weight(self, alpha: float) -> float:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        alpha = alpha * alpha
        return float(
            self.cfg.swing_task_min_weight
            + (self.cfg.swing_task_max_weight - self.cfg.swing_task_min_weight) * alpha
        )

    def _effective_swing_alpha(self, alpha: float) -> float:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        transition = float(self.cfg.swing_liftoff_transition_alpha)
        if alpha <= transition:
            return 0.0
        return float(np.clip((alpha - transition) / max(1.0 - transition, 1e-6), 0.0, 1.0))

    def _contact_flags(self, phase: PhaseState) -> tuple[bool, bool]:
        liftoff_confirmed = bool(phase.metadata.get("liftoff_confirmed", False))
        transition_alpha = float(self.cfg.swing_liftoff_transition_alpha)
        if phase.phase == ContactPhase.LEFT_SWING:
            if not liftoff_confirmed:
                return True, True
            return bool(phase.alpha < transition_alpha), True
        if phase.phase == ContactPhase.RIGHT_SWING:
            if not liftoff_confirmed:
                return True, True
            return True, bool(phase.alpha < transition_alpha)
        return True, True

    def _support_release_alpha(self, phase: PhaseState) -> float:
        if phase.phase == ContactPhase.DOUBLE_SUPPORT:
            return 1.0
        if not bool(phase.metadata.get("liftoff_confirmed", False)):
            return 0.0
        return self._effective_swing_alpha(phase.alpha)

    def _foot_command(
        self,
        side: str,
        phase: PhaseState,
        footsteps: FootstepPlan,
        in_contact: bool,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        pos = np.asarray(
            footsteps.left_position_world if side == "left" else footsteps.right_position_world,
            dtype=np.float64,
        ).copy()
        vel = np.asarray(
            footsteps.left_velocity_world if side == "left" else footsteps.right_velocity_world,
            dtype=np.float64,
        ).copy()
        anchor = np.asarray(
            footsteps.left_anchor_world if side == "left" else footsteps.right_anchor_world,
            dtype=np.float64,
        ).copy()
        if in_contact:
            return anchor, np.zeros(3, dtype=np.float64), 1.0
        swing_alpha = self._effective_swing_alpha(phase.alpha)
        smooth_pos = anchor + swing_alpha * (pos - anchor)
        smooth_vel = vel * swing_alpha
        return smooth_pos, smooth_vel, self._swing_task_weight(swing_alpha)

    def _pelvis_velocity_command(
        self,
        frame: CorridorPlanFrame,
        intent: TraversalIntent,
        phase: PhaseState,
        gap_severity: float,
    ) -> np.ndarray:
        vel_xy = np.asarray(intent.planar_velocity, dtype=np.float64).copy()
        if phase.phase == ContactPhase.DOUBLE_SUPPORT:
            vel_xy *= float(self.cfg.double_support_velocity_scale)
        elif not bool(phase.metadata.get("liftoff_confirmed", False)):
            vel_xy[:] = 0.0
        else:
            release = self._support_release_alpha(phase)
            single_support_scale = float(
                self.cfg.single_support_velocity_scale
                + (self.cfg.through_gap_single_support_velocity_scale - self.cfg.single_support_velocity_scale)
                * float(np.clip(gap_severity, 0.0, 1.0))
            )
            vel_xy *= (0.2 + 0.8 * release) * single_support_scale
        return np.asarray([vel_xy[0], vel_xy[1], frame.h_dot], dtype=np.float64)

    def _clamp_pelvis_xy_to_support(
        self,
        intent: TraversalIntent,
        phase: PhaseState,
        footsteps: FootstepPlan,
        left_in_contact: bool,
        right_in_contact: bool,
    ) -> np.ndarray:
        envelope_state = self.envelope.evaluate(intent)
        yaw = float(intent.yaw)
        rot = np.asarray(
            [
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)],
            ],
            dtype=np.float64,
        )
        support_points = []
        if left_in_contact:
            support_points.append(np.asarray(footsteps.left_position_world[:2], dtype=np.float64))
        if right_in_contact:
            support_points.append(np.asarray(footsteps.right_position_world[:2], dtype=np.float64))
        if not support_points:
            support_points = [
                np.asarray(footsteps.left_position_world[:2], dtype=np.float64),
                np.asarray(footsteps.right_position_world[:2], dtype=np.float64),
            ]
        support_mid = np.mean(np.stack(support_points, axis=0), axis=0)
        single_support = len(support_points) == 1
        desired = np.asarray(intent.planar_position, dtype=np.float64)
        local_offset = rot.T @ (desired - support_mid)
        max_forward = self.cfg.single_support_forward_offset if single_support else self.cfg.max_pelvis_forward_offset
        max_backward = self.cfg.single_support_backward_offset if single_support else self.cfg.max_pelvis_backward_offset
        max_lateral = self.cfg.single_support_lateral_offset if single_support else self.cfg.max_pelvis_lateral_offset
        if phase.phase != ContactPhase.DOUBLE_SUPPORT and not bool(phase.metadata.get("liftoff_confirmed", False)):
            max_forward = float(self.cfg.pre_liftoff_forward_offset)
            max_backward = float(self.cfg.pre_liftoff_backward_offset)
            max_lateral = float(self.cfg.pre_liftoff_lateral_offset)
        if single_support:
            release = self._support_release_alpha(phase)
            release_scale = 0.25 + 0.75 * release
            max_forward *= release_scale
            max_backward *= release_scale
            max_lateral *= release_scale
        local_offset[0] = float(
            np.clip(
                local_offset[0],
                -max_backward,
                max_forward,
            )
        )
        local_offset[1] = float(
            np.clip(
                local_offset[1],
                -max_lateral
                * (1.0 - (1.0 - float(self.cfg.through_gap_pelvis_lateral_scale)) * envelope_state.gap_severity),
                max_lateral
                * (1.0 - (1.0 - float(self.cfg.through_gap_pelvis_lateral_scale)) * envelope_state.gap_severity),
            )
        )
        return support_mid + rot @ local_offset
