"""
Humanoid environment plugins for experiment framework.

Provides MJX-based envs for MBD/2GO planning with obstacle support.
"""

from typing import Dict, Any
import numpy as np

from genedynamics.envs.factories import make_env, make_energy
from ...framework.base import EnvironmentPlugin


class HumanoidSimplifiedMjxPlugin(EnvironmentPlugin):
    """
    Plugin for gymnasium humanoid with MJX (JAX physics).
    State: flat [qpos; qvel], nq=24, nv=23. Position: state[:3].
    """

    @property
    def name(self) -> str:
        return "humanoid_simplified_mjx"

    def create_env(self, config: Dict[str, Any]) -> Any:
        env_kw = {k: v for k, v in config.items() if k != 'physics_backend'}
        return make_env(self.name, **env_kw)

    def create_energy(self) -> Any:
        return make_energy(self.name)

    def get_state_dim(self) -> int:
        return 47

    def extract_position(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=np.float32)[:3]

    def get_position_dim(self) -> int:
        return 3


class HumanoidG1MjxPlugin(EnvironmentPlugin):
    """
    Plugin for Unitree G1 with MJX (mujoco_menagerie).
    """

    @property
    def name(self) -> str:
        return "humanoid_g1_mjx"

    def create_env(self, config: Dict[str, Any]) -> Any:
        env_kw = {k: v for k, v in config.items() if k != 'physics_backend'}
        return make_env(self.name, **env_kw)

    def create_energy(self) -> Any:
        return make_energy(self.name)

    def get_state_dim(self) -> int:
        return 71

    def extract_position(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=np.float32)[:3]

    def get_position_dim(self) -> int:
        return 3


class HumanoidRunBraxPlugin(EnvironmentPlugin):
    """Plugin for humanoid run with Brax positional physics (MBD-style)."""

    @property
    def name(self) -> str:
        return "humanoid_run_brax"

    def create_env(self, config: Dict[str, Any]) -> Any:
        env_kw = {k: v for k, v in config.items() if k != 'physics_backend'}
        return make_env(self.name, **env_kw)

    def create_energy(self) -> Any:
        return make_energy(self.name)

    def get_state_dim(self) -> int:
        return 47

    def extract_position(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=np.float32)[:3]

    def get_position_dim(self) -> int:
        return 3
