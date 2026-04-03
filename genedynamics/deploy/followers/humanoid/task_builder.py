"""
Build humanoid task specs from footsteps, contact, and upper-body intent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genedynamics.deploy.followers.common.plan_schema import CorridorPlanFrame
from genedynamics.deploy.followers.common.traversal_intent import TraversalIntent
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
    body_pitch_from_crouch_gain: float = 0.08


class HumanoidTaskBuilder:
    def __init__(self, cfg: HumanoidTaskBuilderConfig | None = None) -> None:
        self.cfg = cfg or HumanoidTaskBuilderConfig()

    def build(
        self,
        frame: CorridorPlanFrame,
        intent: TraversalIntent,
        phase: PhaseState,
        footsteps: FootstepPlan,
        upper_body: UpperBodyTargets,
    ) -> HumanoidTaskSpec:
        pelvis_xy = self._clamp_pelvis_xy_to_support(intent, footsteps)
        pelvis = PelvisTask(
            position_world=np.asarray([pelvis_xy[0], pelvis_xy[1], intent.body_height], dtype=np.float64),
            yaw_world=float(intent.yaw),
            roll_world=0.0,
            pitch_world=float(self.cfg.body_pitch_from_crouch_gain) * float(upper_body.crouch_ratio),
        )
        return HumanoidTaskSpec(
            plan_frame=frame,
            pelvis=pelvis,
            torso_yaw=float(upper_body.torso_yaw),
            left_foot=FootTask(
                position_world=np.asarray(footsteps.left_position_world, dtype=np.float64).copy(),
                velocity_world=np.asarray(footsteps.left_velocity_world, dtype=np.float64).copy(),
                yaw_world=float(intent.yaw),
                in_contact=phase.phase != ContactPhase.LEFT_SWING,
                weight=1.0 if phase.phase != ContactPhase.LEFT_SWING else 0.35,
            ),
            right_foot=FootTask(
                position_world=np.asarray(footsteps.right_position_world, dtype=np.float64).copy(),
                velocity_world=np.asarray(footsteps.right_velocity_world, dtype=np.float64).copy(),
                yaw_world=float(intent.yaw),
                in_contact=phase.phase != ContactPhase.RIGHT_SWING,
                weight=1.0 if phase.phase != ContactPhase.RIGHT_SWING else 0.35,
            ),
            left_arm=upper_body.left_arm,
            right_arm=upper_body.right_arm,
            joint_hints=dict(upper_body.joint_hints),
            extras={
                "crouch_ratio": float(upper_body.crouch_ratio),
                "phase": phase.phase.value,
                "step_index": phase.step_index,
                "footsteps": dict(footsteps.metadata),
            },
        )

    def _clamp_pelvis_xy_to_support(self, intent: TraversalIntent, footsteps: FootstepPlan) -> np.ndarray:
        yaw = float(intent.yaw)
        rot = np.asarray(
            [
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)],
            ],
            dtype=np.float64,
        )
        support_mid = 0.5 * (
            np.asarray(footsteps.left_position_world[:2], dtype=np.float64)
            + np.asarray(footsteps.right_position_world[:2], dtype=np.float64)
        )
        desired = np.asarray(intent.planar_position, dtype=np.float64)
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
