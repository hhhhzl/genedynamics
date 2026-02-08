"""
D3IL Avoiding 9D environment plugin (state [tcp_xy, q], action qdot).
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np

from ...framework.base import EnvironmentPlugin
from enerdynamics.envs.external.d3il import D3ILAvoiding7dVelEnv, D3ILAvoiding7dVelConfig


class D3ILAvoiding9DPlugin(EnvironmentPlugin):
    """
    Environment plugin for D3IL Avoiding 9D: 7D velocity action, 9D state [tcp_xy, q].
    """

    @property
    def name(self) -> str:
        return "d3il_avoiding_9d"

    def create_env(self, config: Dict[str, Any]) -> Any:
        env_cfg = D3ILAvoiding7dVelConfig(
            render=bool(config.get("render", False)),
            obstacle_level=config.get("obstacle_level"),
            obstacle_radius_by_level=config.get("obstacle_radius_by_level"),
            obstacles=config.get("obstacles"),
            robot_radius=config.get("robot_radius"),
            collision_ee_only=bool(config.get("collision_ee_only", False)),
        )
        env = D3ILAvoiding7dVelEnv(env_cfg)
        if "horizon" in config:
            env.horizon = int(config["horizon"])
        if "dt" in config:
            env.dt = float(config["dt"])
        return env

    def create_energy(self) -> Any:
        from enerdynamics.core.energy import LegacyEnergyFunctional, EnergyTerm
        import jax.numpy as jnp

        goal_xy = jnp.array([0.5, 0.35], dtype=jnp.float32)  # center of last obstacle row

        def task_energy(x, u, ctx):
            _ = (u, ctx)
            pos = x[:2]
            return jnp.sum((pos - goal_xy) ** 2)

        return LegacyEnergyFunctional({"task": EnergyTerm(task_energy, 1.0)})

    def get_state_dim(self) -> int:
        return 9

    def extract_position(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        if s.size >= 2:
            return s[:2]
        return s[: min(2, s.size)]
