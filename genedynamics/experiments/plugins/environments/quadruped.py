"""
Quadruped environment plugins for experiment framework.

Provides MJX-based envs for MBD/2GO planning with obstacle support.
"""

from typing import Dict, Any
import numpy as np

from genedynamics.envs.factories import make_env, make_energy
from ...framework.base import EnvironmentPlugin


class QuadrupedFlatMjxPlugin(EnvironmentPlugin):
    """
    Plugin for quadruped ant with MJX (JAX physics).
    State: flat [qpos; qvel], nq=15, nv=14. Position: state[:3].
    """

    @property
    def name(self) -> str:
        return "quadruped_flat_mjx"

    def create_env(self, config: Dict[str, Any]) -> Any:
        env_kw = {k: v for k, v in config.items() if k != 'physics_backend'}
        return make_env(self.name, **env_kw)

    def create_energy(self) -> Any:
        return make_energy(self.name)

    def get_state_dim(self) -> int:
        return 29

    def extract_position(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=np.float32)[:3]

    def get_position_dim(self) -> int:
        return 3


class QuadrupedGo2MjxPlugin(EnvironmentPlugin):
    """
    Plugin for Unitree Go2 with MJX (mujoco_menagerie).
    """

    @property
    def name(self) -> str:
        return "quadruped_go2_mjx"

    def create_env(self, config: Dict[str, Any]) -> Any:
        env_kw = {k: v for k, v in config.items() if k != 'physics_backend'}
        return make_env(self.name, **env_kw)

    def create_energy(self) -> Any:
        return make_energy(self.name)

    def get_state_dim(self) -> int:
        return 37

    def extract_position(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=np.float32)[:3]

    def get_position_dim(self) -> int:
        return 3
