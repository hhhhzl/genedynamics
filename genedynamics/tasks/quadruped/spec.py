"""
Quadruped task spec for flat/rough/push tasks.

State: [qpos; qvel] - base pose (3 pos + 4 quat or 6 euler) + joint angles, velocities
Position: base xyz (state[:3] or extracted from quat)
Control: joint torques (sim) or joint position targets (real, position mode)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


class QuadrupedTaskSpec:
    """
    Task spec for quadruped locomotion (flat, rough, push).

    - State layout: [qpos; qvel] - MuJoCo ant: nq=15 (3+4+8), nv=14
    - Position: base xyz (first 3 of qpos for free joint)
    - Control: 8D torque (sim) or 8D joint targets (real position mode)
    """

    # Ant-like default: 3 pos + 4 quat + 8 joints = 15, vel 6+8=14
    NQ_DEFAULT = 15
    NV_DEFAULT = 14
    ACT_DIM_DEFAULT = 8
    POSITION_DIM = 3

    def __init__(
        self,
        nq: int = NQ_DEFAULT,
        nv: int = NV_DEFAULT,
        act_dim: int = ACT_DIM_DEFAULT,
        target: Tuple[float, float, float] = (2.0, 0.0, 0.5),
        control_limit: float = 1.0,
        p_max: float = 5.0,
        v_max: float = 3.0,
    ):
        self.nq = nq
        self.nv = nv
        self.act_dim = act_dim
        self.target = np.asarray(target, dtype=np.float32)
        self.control_limit = control_limit
        self.p_max = p_max
        self.v_max = v_max

    @property
    def state_dim(self) -> int:
        return self.nq + self.nv

    @property
    def position_dim(self) -> int:
        return self.POSITION_DIM

    def extract_position(self, state: np.ndarray) -> np.ndarray:
        """Extract base 3D position from state (first 3 of qpos)."""
        s = np.asarray(state, dtype=np.float32).ravel()
        if s.size >= self.nq:
            return s[:3].copy()
        return s[:3].copy() if s.size >= 3 else s.copy()

    def success_criterion(
        self,
        final_position: np.ndarray,
        target: Optional[np.ndarray] = None,
        success_margin: float = 0.3,
        **kwargs: Any,
    ) -> bool:
        """Check if task succeeded (reached target region)."""
        t = np.asarray(self.target if target is None else target, dtype=np.float64).ravel()[:3]
        p = np.asarray(final_position, dtype=np.float64).ravel()[:3]
        return bool(np.linalg.norm(p - t) < success_margin)

    def state_to_mujoco(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert flat state to MuJoCo format."""
        s = np.asarray(state, dtype=np.float32).ravel()
        return {
            "qpos": s[: self.nq].copy(),
            "qvel": s[self.nq : self.nq + self.nv].copy(),
        }

    def mujoco_to_state(self, mujoco_state: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert MuJoCo state to flat."""
        qpos = np.asarray(mujoco_state.get("qpos", []), dtype=np.float32)
        qvel = np.asarray(mujoco_state.get("qvel", []), dtype=np.float32)
        return np.concatenate([qpos, qvel]).astype(np.float32)
