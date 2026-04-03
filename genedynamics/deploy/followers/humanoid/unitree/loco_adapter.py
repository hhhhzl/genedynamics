"""
Adapter from traversal intent to Unitree-style lower-body locomotion commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from genedynamics.deploy.followers.common.controller_output import (
    LocoCommand,
    MixedHumanoidCommand,
)
from genedynamics.deploy.followers.common.traversal_intent import TraversalIntent


@dataclass
class G1LocoAdapterConfig:
    max_linear_speed_xy: float = 0.75
    max_yaw_rate: float = 1.25
    body_height_nominal: float = 0.75
    body_height_min: float = 0.55
    body_height_max: float = 0.90
    torso_yaw_gain: float = 1.0
    arm_tuck_elbow_gain: float = 1.10
    arm_tuck_shoulder_roll_gain: float = 0.55
    arm_posture_pitch_gain: float = 0.45
    arm_posture_yaw_gain: float = 0.35


class G1LocoAdapter:
    """
    Real-robot adapter for the eventual Unitree loco path.

    The lower body stays high-level while torso/arms remain explicit joint
    references so we can preserve corridor-specific upper-body semantics.
    """

    def __init__(self, cfg: G1LocoAdapterConfig | None = None) -> None:
        self.cfg = cfg or G1LocoAdapterConfig()

    def adapt(self, intent: TraversalIntent) -> MixedHumanoidCommand:
        v_xy = np.asarray(intent.planar_velocity, dtype=np.float64).reshape(2)
        speed = float(np.linalg.norm(v_xy))
        if speed > float(self.cfg.max_linear_speed_xy) and speed > 1e-6:
            v_xy = v_xy * (float(self.cfg.max_linear_speed_xy) / speed)
        yaw_rate = float(np.clip(intent.yaw_rate, -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate))
        body_height = float(
            np.clip(intent.body_height, self.cfg.body_height_min, self.cfg.body_height_max)
        )
        upper_body = self._upper_body_joint_targets(intent)
        return MixedHumanoidCommand(
            lower_body=LocoCommand(
                linear_velocity_xy=v_xy,
                yaw_rate=yaw_rate,
                body_height=body_height,
                gait_mode="walk",
                metadata={"adapter": "g1_loco_adapter"},
            ),
            upper_body_joint_targets=upper_body,
            metadata={"intent_time_sec": float(intent.time_sec)},
        )

    def _upper_body_joint_targets(self, intent: TraversalIntent) -> Dict[str, float]:
        return {
            "waist_yaw_joint": float(self.cfg.torso_yaw_gain) * float(intent.torso_yaw),
            **self._arm_joint_targets("left", intent.left_arm_tuck, intent.left_arm_posture),
            **self._arm_joint_targets("right", intent.right_arm_tuck, intent.right_arm_posture),
        }

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
