"""
Offline planning environment plugin for D3IL Avoiding (no MuJoCo execution).

This exposes `AvoidingPlanEnv` to the ExperimentRunner so you can run:
  env_name: avoiding_plan
  method: edoc / mbd

and get an offline/open-loop trajectory.
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np

from ...framework.base import EnvironmentPlugin
from genedynamics.envs.external.d3il.avoiding_plan_env import AvoidingPlanEnv, AvoidingPlanSpec


class AvoidingPlanEnvironmentPlugin(EnvironmentPlugin):
    @property
    def name(self) -> str:
        return "avoiding_plan"

    def create_env(self, config: Dict[str, Any]) -> Any:
        spec = AvoidingPlanSpec(
            dt=float(config.get("dt", 0.035)),
            horizon=int(config.get("horizon", 20)),
            control_limit=float(config.get("control_limit", config.get("action_limit", 0.05))),
            target_xy=tuple(config.get("target_xy", (0.4, 0.35))),
        )
        env = AvoidingPlanEnv(spec)

        # Optional: bounds used by ExperimentRunner start generation (fallback exists)
        env.p_max = float(config.get("p_max", 2.0))
        return env

    def create_energy(self) -> Any:
        """
        Default energy: quadratic distance to goal on measured xy (dims 2:4).
        """
        from genedynamics.core.energy import LegacyEnergyFunctional, EnergyTerm
        import jax.numpy as jnp

        goal_xy = jnp.array([0.4, 0.35], dtype=jnp.float32)

        def task_energy(x, u, ctx):
            _ = (u, ctx)
            pos = x[2:4]
            return jnp.sum((pos - goal_xy) ** 2)

        return LegacyEnergyFunctional({"task": EnergyTerm(task_energy, 1.0)})

    def get_state_dim(self) -> int:
        return 4

    def extract_position(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        return s[2:4] if s.size >= 4 else s[:2]


