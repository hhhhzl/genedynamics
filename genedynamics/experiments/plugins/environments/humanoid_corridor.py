"""
Humanoid corridor 2D environment plugin for experiment framework.

Provides the planning-level environment for G1 narrow corridor
obstacle avoidance (14D state, 9D action, velocity-rate dynamics).
"""

from typing import Dict, Any
import numpy as np

from genedynamics.envs.factories import make_env, make_energy
from ...framework.base import EnvironmentPlugin


class HumanoidCorridor2DPlugin(EnvironmentPlugin):
    """
    Plugin for humanoid corridor obstacle avoidance (2D planning).

    State (14D): [x, y, psi, h, psi_torso, a_L, a_R, p_L, p_R,
                  v_x, v_y, omega, h_dot, psi_dot_torso]
    Action (9D): velocity/rate commands.
    """

    @property
    def name(self) -> str:
        return "humanoid_corridor_2d"

    def create_env(self, config: Dict[str, Any]) -> Any:
        env_kw = {k: v for k, v in config.items() if k != "physics_backend"}
        return make_env(self.name, **env_kw)

    def create_energy(self, env: Any = None) -> Any:
        # Bind the energy to the selected zone/long scene. The generic factory
        # constructs the default medium scene, which gives zone experiments the
        # wrong goal and corridor geometry.
        if env is not None:
            from genedynamics.envs.domains.humanoid.corridor import make_corridor_energy

            return make_corridor_energy(env)
        return make_energy(self.name)

    def get_state_dim(self) -> int:
        return 14

    def extract_position(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=np.float32)[:2]

    def get_position_dim(self) -> int:
        return 2
