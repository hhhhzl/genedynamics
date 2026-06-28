"""
Upper-body task mapping for humanoid corridor traversal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from genedynamics.deploy.followers.common.plan_schema import CorridorPlanFrame
from genedynamics.deploy.followers.common.traversal_intent import TraversalIntent
from genedynamics.deploy.followers.humanoid.corridor_envelope import (
    CorridorEnvelopeConfig,
    CorridorEnvelopeHeuristics,
)
from genedynamics.deploy.followers.humanoid.task_spec import ArmJointTask


@dataclass
class HumanoidUpperBodyMapperConfig:
    torso_yaw_gain: float = 1.0
    body_pitch_from_crouch_gain: float = 0.08
    body_height_nominal: float = 0.75
    body_height_min: float = 0.55
    arm_tuck_elbow_gain: float = 1.10
    # Lowered 0.55 -> 0.15 so the arms abduct less; combined with arm_elbow_base
    # this keeps the wrists drawn in toward the body so they don't swing into
    # side-wall obstacles during execution (the planner is fixed). Tuned on
    # zone_c (audit_exec_collision): -0.056 (collide) -> +0.060 (clear).
    arm_tuck_shoulder_roll_gain: float = 0.15
    # Baseline elbow flexion applied even when the plan does not command a tuck
    # (a=0), drawing the wrists in. 0.60 rad from the same zone_c tuning.
    arm_elbow_base: float = 0.60
    arm_posture_pitch_gain: float = 0.45
    arm_posture_yaw_gain: float = 0.35
    through_gap_shoulder_roll_scale: float = 0.72
    through_gap_elbow_scale: float = 0.55
    envelope: CorridorEnvelopeConfig = field(default_factory=CorridorEnvelopeConfig)


@dataclass
class UpperBodyTargets:
    torso_yaw: float
    left_arm: ArmJointTask
    right_arm: ArmJointTask
    joint_hints: Dict[str, float]
    crouch_ratio: float
    narrowness: float
    gap_severity: float
    effective_width: float
    through_gap_mode: bool


class HumanoidUpperBodyMapper:
    def __init__(self, cfg: HumanoidUpperBodyMapperConfig | None = None) -> None:
        self.cfg = cfg or HumanoidUpperBodyMapperConfig()
        self.envelope = CorridorEnvelopeHeuristics(self.cfg.envelope)

    def map(self, frame: CorridorPlanFrame) -> UpperBodyTargets:
        intent = TraversalIntent.from_plan_frame(frame)
        envelope = self.envelope.evaluate(intent)
        left_arm = ArmJointTask(
            joint_targets=self._arm_joint_targets("left", frame.a_left, frame.p_left, envelope.gap_severity)
        )
        right_arm = ArmJointTask(
            joint_targets=self._arm_joint_targets("right", frame.a_right, frame.p_right, envelope.gap_severity)
        )
        joint_hints: Dict[str, float] = {
            "waist_yaw_joint": float(self.cfg.torso_yaw_gain) * float(frame.psi_torso),
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": float(self.cfg.body_pitch_from_crouch_gain) * envelope.crouch_ratio,
        }
        joint_hints.update(left_arm.joint_targets)
        joint_hints.update(right_arm.joint_targets)
        return UpperBodyTargets(
            torso_yaw=joint_hints["waist_yaw_joint"],
            left_arm=left_arm,
            right_arm=right_arm,
            joint_hints=joint_hints,
            crouch_ratio=envelope.crouch_ratio,
            narrowness=envelope.narrowness,
            gap_severity=envelope.gap_severity,
            effective_width=envelope.effective_width,
            through_gap_mode=envelope.through_gap_mode,
        )

    def _arm_joint_targets(self, side: str, tuck: float, posture: float, narrowness: float) -> Dict[str, float]:
        a = float(np.clip(tuck, 0.0, 1.0))
        p = float(np.clip(posture, -1.0, 1.0))
        roll_sign = 1.0 if side == "left" else -1.0
        prefix = "left" if side == "left" else "right"
        shoulder_roll = float(self.cfg.arm_tuck_shoulder_roll_gain) * (1.0 - a)
        shoulder_roll *= float(1.0 - self.cfg.through_gap_shoulder_roll_scale * narrowness)
        elbow_target = (float(self.cfg.arm_elbow_base)
                        + float(self.cfg.arm_tuck_elbow_gain) * a
                        + float(self.cfg.through_gap_elbow_scale) * narrowness)
        return {
            f"{prefix}_shoulder_pitch_joint": float(self.cfg.arm_posture_pitch_gain) * p,
            f"{prefix}_shoulder_roll_joint": roll_sign * shoulder_roll,
            f"{prefix}_shoulder_yaw_joint": roll_sign * float(self.cfg.arm_posture_yaw_gain) * p,
            f"{prefix}_elbow_joint": elbow_target,
            f"{prefix}_wrist_roll_joint": 0.0,
            f"{prefix}_wrist_pitch_joint": 0.0,
            f"{prefix}_wrist_yaw_joint": 0.0,
        }

    def _crouch_ratio(self, height: float) -> float:
        denom = max(self.cfg.body_height_nominal - self.cfg.body_height_min, 1e-6)
        return float(np.clip((self.cfg.body_height_nominal - float(height)) / denom, 0.0, 1.0))
