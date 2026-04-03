"""
Upper-body task mapping for humanoid corridor traversal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from genedynamics.deploy.followers.common.plan_schema import CorridorPlanFrame
from genedynamics.deploy.followers.humanoid.task_spec import ArmJointTask


@dataclass
class HumanoidUpperBodyMapperConfig:
    torso_yaw_gain: float = 1.0
    body_pitch_from_crouch_gain: float = 0.08
    body_height_nominal: float = 0.75
    body_height_min: float = 0.55
    arm_tuck_elbow_gain: float = 1.10
    arm_tuck_shoulder_roll_gain: float = 0.55
    arm_posture_pitch_gain: float = 0.45
    arm_posture_yaw_gain: float = 0.35


@dataclass
class UpperBodyTargets:
    torso_yaw: float
    left_arm: ArmJointTask
    right_arm: ArmJointTask
    joint_hints: Dict[str, float]
    crouch_ratio: float


class HumanoidUpperBodyMapper:
    def __init__(self, cfg: HumanoidUpperBodyMapperConfig | None = None) -> None:
        self.cfg = cfg or HumanoidUpperBodyMapperConfig()

    def map(self, frame: CorridorPlanFrame) -> UpperBodyTargets:
        crouch_ratio = self._crouch_ratio(frame.h)
        left_arm = ArmJointTask(joint_targets=self._arm_joint_targets("left", frame.a_left, frame.p_left))
        right_arm = ArmJointTask(joint_targets=self._arm_joint_targets("right", frame.a_right, frame.p_right))
        joint_hints: Dict[str, float] = {
            "waist_yaw_joint": float(self.cfg.torso_yaw_gain) * float(frame.psi_torso),
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": float(self.cfg.body_pitch_from_crouch_gain) * crouch_ratio,
        }
        joint_hints.update(left_arm.joint_targets)
        joint_hints.update(right_arm.joint_targets)
        return UpperBodyTargets(
            torso_yaw=joint_hints["waist_yaw_joint"],
            left_arm=left_arm,
            right_arm=right_arm,
            joint_hints=joint_hints,
            crouch_ratio=crouch_ratio,
        )

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
