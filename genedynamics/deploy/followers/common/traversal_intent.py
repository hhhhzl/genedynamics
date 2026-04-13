"""
Robot-agnostic traversal intent derived from corridor plans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from genedynamics.deploy.followers.common.plan_schema import CorridorPlanFrame


@dataclass
class TraversalIntent:
    """
    High-level traversal intent shared across embodiments.

    This keeps the planner semantics reusable while letting each robot family
    interpret the intent in its own task space.
    """

    time_sec: float
    planar_position: np.ndarray
    yaw: float
    body_height: float
    planar_velocity: np.ndarray
    yaw_rate: float
    torso_yaw: float = 0.0
    torso_yaw_rate: float = 0.0
    left_arm_tuck: float = 0.0
    right_arm_tuck: float = 0.0
    left_arm_posture: float = 0.0
    right_arm_posture: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)
    raw_state: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64), repr=False)

    @property
    def x(self) -> float:
        return float(self.planar_position[0])

    @property
    def y(self) -> float:
        return float(self.planar_position[1])

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.planar_velocity))

    @classmethod
    def from_plan_frame(cls, frame: CorridorPlanFrame) -> "TraversalIntent":
        return cls(
            time_sec=float(frame.time_sec),
            planar_position=np.asarray([frame.x, frame.y], dtype=np.float64),
            yaw=float(frame.psi),
            body_height=float(frame.h),
            planar_velocity=np.asarray([frame.v_x, frame.v_y], dtype=np.float64),
            yaw_rate=float(frame.omega),
            torso_yaw=float(frame.psi_torso),
            torso_yaw_rate=float(frame.psi_dot_torso),
            left_arm_tuck=float(frame.a_left),
            right_arm_tuck=float(frame.a_right),
            left_arm_posture=float(frame.p_left),
            right_arm_posture=float(frame.p_right),
            extras=dict(frame.extras),
            raw_state=frame.raw_state.copy(),
        )
