"""
Humanoid task spec (simplified).

State: [qpos; qvel] - MuJoCo humanoid: nq=24 (2+1+4+17), nv=23
Position: base xyz (first 3 of qpos)
Control: 17D torque (sim) or 17D joint targets (real position mode)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


class HumanoidTaskSpec:
    """
    Task spec for simplified humanoid locomotion.

    - State layout: [qpos; qvel] - humanoid: nq=24, nv=23
    - Position: base xyz (first 3 of qpos)
    - Control: 17D torque (sim) or joint targets (real)
    """

    NQ_DEFAULT = 24
    NV_DEFAULT = 23
    ACT_DIM_DEFAULT = 17
    POSITION_DIM = 3

    def __init__(
        self,
        nq: int = NQ_DEFAULT,
        nv: int = NV_DEFAULT,
        act_dim: int = ACT_DIM_DEFAULT,
        target: Tuple[float, float, float] = (3.0, 0.0, 1.0),
        control_limit: float = 0.4,
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
        """Extract base 3D position (x,y,z from qpos)."""
        s = np.asarray(state, dtype=np.float32).ravel()
        if s.size >= self.nq:
            return s[:3].copy()
        return s[:3].copy() if s.size >= 3 else s.copy()

    def success_criterion(
        self,
        final_position: np.ndarray,
        target: Optional[np.ndarray] = None,
        success_margin: float = 0.4,
        **kwargs: Any,
    ) -> bool:
        """Check if task succeeded."""
        t = np.asarray(target or self.target, dtype=np.float64).ravel()[:3]
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
