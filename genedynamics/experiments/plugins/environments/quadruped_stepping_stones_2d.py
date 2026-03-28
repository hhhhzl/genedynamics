"""
Quadruped stepping-stones 2D environment plugin.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from genedynamics.envs.quadruped_stepping_stones_2d import (
    QuadrupedSteppingStones2DEnv,
    make_stepping_stones_energy,
)
from ...framework.base import EnvironmentPlugin


class QuadrupedSteppingStones2DPlugin(EnvironmentPlugin):
    @property
    def name(self) -> str:
        return "quadruped_stepping_stones_2d"

    def create_env(self, config: Dict[str, Any]) -> Any:
        return QuadrupedSteppingStones2DEnv(**config)

    def create_energy(self, env: Any = None) -> Any:
        env_obj = env if isinstance(env, QuadrupedSteppingStones2DEnv) else QuadrupedSteppingStones2DEnv()
        return make_stepping_stones_energy(env_obj)

    def get_state_dim(self) -> int:
        return 4

    def extract_position(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        if s.size >= 4:
            return np.array([0.5 * (s[0] + s[2]), 0.5 * (s[1] + s[3])], dtype=np.float32)
        if s.size >= 2:
            return s[:2]
        return np.pad(s, (0, max(0, 2 - s.size)), constant_values=0.0)[:2]

    def get_position_dim(self) -> int:
        return 2

