"""
UAV 3D obstacle navigation task spec.

State: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz] (12D)
Control: [T1, T2, T3, T4] (4 motor thrusts)
Constraints: collision avoidance, control saturation, optional tilt limits
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from genedynamics.envs.utils.state_converter import state_12d_to_mujoco, mujoco_to_state_12d


class UAV3DTaskSpec:
    """
    Task spec for UAV 3D obstacle navigation.

    - State layout: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
    - Position: state[:3]
    - Control: 4 motor thrusts [0, control_limit]
    """

    STATE_DIM = 12
    ACT_DIM = 4
    NQ = 7  # MuJoCo qpos: 3 pos + 4 quat
    NV = 6  # MuJoCo qvel: 3 vel + 3 ang_vel

    def __init__(
        self,
        target: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        control_limit: float = 1.0,
        p_max: float = 2.0,
        v_max: float = 2.0,
    ):
        self.target = np.asarray(target, dtype=np.float32)
        self.control_limit = control_limit
        self.p_max = p_max
        self.v_max = v_max

    def extract_position(self, state: np.ndarray) -> np.ndarray:
        """Extract 3D position from state."""
        s = np.asarray(state, dtype=np.float32).ravel()
        return s[:3].copy() if s.size >= 3 else s.copy()

    def success_criterion(
        self,
        final_position: np.ndarray,
        target: Optional[np.ndarray] = None,
        success_margin: float = 0.15,
        **kwargs: Any,
    ) -> bool:
        """Check if task succeeded."""
        t = np.asarray(target or self.target, dtype=np.float64).ravel()[:3]
        p = np.asarray(final_position, dtype=np.float64).ravel()[:3]
        return bool(np.linalg.norm(p - t) < success_margin)

    def state_to_mujoco(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert 12D state to MuJoCo format."""
        return state_12d_to_mujoco(np.asarray(state, dtype=np.float32))

    def mujoco_to_state(self, mujoco_state: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert MuJoCo state to 12D."""
        return mujoco_to_state_12d(mujoco_state)
