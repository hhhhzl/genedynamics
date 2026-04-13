"""
Humanoid-specific task objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np

from genedynamics.deploy.followers.common.plan_schema import CorridorPlanFrame


class ContactPhase(str, Enum):
    DOUBLE_SUPPORT = "double_support"
    LEFT_SWING = "left_swing"
    RIGHT_SWING = "right_swing"


@dataclass
class PhaseState:
    phase: ContactPhase
    alpha: float
    step_index: int
    swing_foot: Optional[str]
    support_feet: Tuple[str, ...]
    phase_elapsed: float
    phase_duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PelvisTask:
    position_world: np.ndarray
    yaw_world: float
    roll_world: float = 0.0
    pitch_world: float = 0.0
    linear_velocity_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    angular_velocity_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))


@dataclass
class FootTask:
    position_world: np.ndarray
    velocity_world: np.ndarray
    yaw_world: float
    in_contact: bool
    roll_world: float = 0.0
    pitch_world: float = 0.0
    angular_velocity_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    weight: float = 1.0


@dataclass
class ArmJointTask:
    joint_targets: Dict[str, float]


@dataclass
class FootContactObservation:
    position_world: np.ndarray
    velocity_world: np.ndarray
    rotation_world: np.ndarray
    angular_velocity_world: np.ndarray
    in_contact: bool
    contact_count: int = 0
    support_load: float = 0.0


@dataclass
class ContactObservations:
    left: FootContactObservation
    right: FootContactObservation


@dataclass
class HumanoidTaskSpec:
    plan_frame: CorridorPlanFrame
    pelvis: PelvisTask
    torso_yaw: float
    left_foot: FootTask
    right_foot: FootTask
    left_arm: ArmJointTask
    right_arm: ArmJointTask
    joint_hints: Dict[str, float] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


# Backward-compatible alias while the old package is still in service.
FollowerTasks = HumanoidTaskSpec
