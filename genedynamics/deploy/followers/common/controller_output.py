"""
Controller-side output objects shared by follower backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class JointTargets:
    q_ref: np.ndarray
    qd_ref: Optional[np.ndarray] = None
    tau_ff: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocoCommand:
    linear_velocity_xy: np.ndarray
    yaw_rate: float
    body_height: Optional[float] = None
    gait_mode: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MixedHumanoidCommand:
    lower_body: Optional[LocoCommand] = None
    upper_body_joint_targets: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
