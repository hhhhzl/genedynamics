"""
Energy functional registrations for environments.

This module registers energy functionals for all environments to the energy registry.
This centralizes energy registration and allows users to extend with custom energy functions.
"""

import jax.numpy as jnp

try:
    from enerdynamics.core.registry.energy import register_energy
    from enerdynamics.core.energy import LegacyEnergyFunctional, EnergyTerm
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    register_energy = None


if REGISTRY_AVAILABLE:
    
    def _make_double_integrator_box_energy():
        """Energy functional for double_integrator_box environment."""
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
    
    register_energy("double_integrator_box", _make_double_integrator_box_energy)
    
    
    def _make_double_integrator_box_2d_energy():
        """Energy functional for double_integrator_box_2d environment."""
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
            "task": EnergyTerm(task_energy, 2.0),
            "box": EnergyTerm(box_energy, 1.0),
        })
    
    register_energy("double_integrator_box_2d", _make_double_integrator_box_2d_energy)
    
    
    def _make_single_integrator_box_2d_energy():
        """Energy functional for single_integrator_box_2d environment."""
        def task_energy(x, u, ctx):
            pos = x[:2]
            return jnp.sum((pos - jnp.array([0.0, 0.0], dtype=jnp.float32)) ** 2)
        
        def box_energy(x, u, ctx):
            p_max = 2.0
            pos_violate = jnp.maximum(0.0, jnp.abs(x[:2]) - p_max)
            pen = jnp.sum(pos_violate ** 2)
            return pen
        
        return LegacyEnergyFunctional({
            "task": EnergyTerm(task_energy, 1.0),
            "box": EnergyTerm(box_energy, 1.0),
        })
    
    register_energy("single_integrator_box_2d", _make_single_integrator_box_2d_energy)

