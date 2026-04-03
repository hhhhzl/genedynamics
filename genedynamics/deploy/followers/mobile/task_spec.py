"""
Mobile-base task spec placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class MobileBaseTaskSpec:
    planar_position: np.ndarray
    yaw: float
    velocity: np.ndarray
    extras: Dict[str, Any] = field(default_factory=dict)
