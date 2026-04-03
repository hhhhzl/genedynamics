"""
Quadruped task spec placeholder for future corridor adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class QuadrupedTaskSpec:
    body_position_world: np.ndarray
    body_yaw_world: float
    foot_targets: Dict[str, np.ndarray] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)
