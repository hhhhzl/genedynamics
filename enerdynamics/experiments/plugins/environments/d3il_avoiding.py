"""
D3IL Avoiding environment plugin.
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np

from ...framework.base import EnvironmentPlugin
from enerdynamics.envs.external.d3il import D3ILAvoidingEnv, D3ILAvoidingConfig


class D3ILAvoidingPlugin(EnvironmentPlugin):
    """
    Environment plugin for D3IL Avoiding (MuJoCo).
    """

    @property
    def name(self) -> str:
        return "d3il_avoiding"

    def create_env(self, config: Dict[str, Any]) -> Any:
        # Note: we keep config minimal for now. Future tasks will need task-specific parameters.
        env_cfg = D3ILAvoidingConfig(
            render=bool(config.get("render", False)),
        )
        env = D3ILAvoidingEnv(env_cfg)
        # Allow overriding horizon/dt if desired (mostly for logging/compat)
        if "horizon" in config:
            env.horizon = int(config["horizon"])
        if "dt" in config:
            env.dt = float(config["dt"])
        return env

    def create_energy(self) -> Any:
        """
        Minimal energy functional for compatibility.

        We use a simple 2D goal-seeking energy on the *measured* position (x, y),
        where the D3IL avoiding goal is a line at y ~= 0.35 (see D3IL env logic).
        """
        from enerdynamics.core.energy import LegacyEnergyFunctional, EnergyTerm
        import jax.numpy as jnp

        goal_xy = jnp.array([0.4, 0.35], dtype=jnp.float32)

        def task_energy(x, u, ctx):
            _ = (u, ctx)
            # x = [x_des, y_des, x, y]
            pos = x[2:4]
            return jnp.sum((pos - goal_xy) ** 2)

        return LegacyEnergyFunctional({"task": EnergyTerm(task_energy, 1.0)})

    def get_state_dim(self) -> int:
        # [x_des, y_des, x, y]
        return 4

    def extract_position(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        if s.size >= 4:
            return s[2:4]
        return s[:2]


