"""
Environment and energy factory functions.

This module provides factory functions to create environments and energy
functionals for different environments. This centralizes environment
creation logic.
"""

import jax.numpy as jnp

from enerdynamics.core.energy import LegacyEnergyFunctional, EnergyTerm


def make_env(name: str):
    """
    Factory function to create environments.
    
    Args:
        name: Environment name (e.g., "double_integrator_box", "double_integrator_box_2d")
        
    Returns:
        Environment instance
        
    Raises:
        ValueError: If environment name is not recognized
    """
    if name == "double_integrator_box":
        from enerdynamics.envs.double_integrator_box import DoubleIntegratorBoxEnv
        return DoubleIntegratorBoxEnv()
    elif name == "double_integrator_box_2d":
        from enerdynamics.envs.double_integrator_box_2d import DoubleIntegratorBox2DEnv
        return DoubleIntegratorBox2DEnv()
    else:
        raise ValueError(f"Unknown environment name: {name}")


def make_energy(env_name: str) -> LegacyEnergyFunctional:
    """
    Factory function to create energy functionals for environments.
    
    Args:
        env_name: Environment name
        
    Returns:
        LegacyEnergyFunctional instance
        
    Raises:
        ValueError: If environment name is not recognized
    """
    if env_name == "double_integrator_box":
        def task_energy(x, u, ctx):
            pos = x[0]
            vel = x[1]
            return (pos - 0.0) ** 2 + (vel ** 2)

        def box_energy(x, u, ctx):
            p_max, v_max = 2.0, 2.0
            pos_violate = jnp.maximum(0.0, jnp.abs(x[0]) - p_max)
            vel_violate = jnp.maximum(0.0, jnp.abs(x[1]) - v_max)
            pen = pos_violate ** 2 + vel_violate ** 2
            return pen

        return LegacyEnergyFunctional({
            "task": EnergyTerm(task_energy, 1.0),
            "box": EnergyTerm(box_energy, 1.0),
        })
    elif env_name == "double_integrator_box_2d":
        def task_energy(x, u, ctx):
            pos = x[:2]
            vel = x[2:]
            return jnp.sum((pos - jnp.array([0.0, 0.0], dtype=jnp.float32)) ** 2) + jnp.sum(vel ** 2)

        def box_energy(x, u, ctx):
            p_max, v_max = 2.0, 2.0
            pos_violate = jnp.maximum(0.0, jnp.abs(x[:2]) - p_max)
            vel_violate = jnp.maximum(0.0, jnp.abs(x[2:]) - v_max)
            pen = jnp.sum(pos_violate ** 2 + vel_violate ** 2)
            return pen

        return LegacyEnergyFunctional({
            "task": EnergyTerm(task_energy, 1.0),
            "box": EnergyTerm(box_energy, 1.0),
        })
    else:
        raise ValueError(f"Unknown environment name: {env_name}")

