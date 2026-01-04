"""
Environment and energy factory functions.

This module provides factory functions to create environments and energy
functionals for different environments. This centralizes environment
creation logic.
"""

from typing import Optional, Any

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    # Create a dummy jnp for type hints
    class DummyJNP:
        @staticmethod
        def array(*args, **kwargs):
            raise RuntimeError("JAX is required but not installed")
        @staticmethod
        def maximum(*args, **kwargs):
            raise RuntimeError("JAX is required but not installed")
        @staticmethod
        def abs(*args, **kwargs):
            raise RuntimeError("JAX is required but not installed")
        @staticmethod
        def sum(*args, **kwargs):
            raise RuntimeError("JAX is required but not installed")
    jnp = DummyJNP()

from enerdynamics.core.energy import LegacyEnergyFunctional, EnergyTerm


def make_env(name: str, **kwargs):
    """
    Factory function to create environments using the registry system.
    
    Args:
        name: Environment name (e.g., "double_integrator_box", "double_integrator_box_2d")
        **kwargs: Additional environment parameters
        
    Returns:
        Environment instance
        
    Raises:
        ValueError: If environment name is not recognized
    """
    # Try to use registry first
    try:
        from enerdynamics.core.registry.environments import get_environment_registry
        registry = get_environment_registry()
        env_class = registry.get_class(name)
        if env_class is not None:
            return registry.create(name, **kwargs)
    except (ImportError, AttributeError):
        # Registry not available, fall back to legacy hardcoded logic
        pass
    
    # Fallback to legacy hardcoded logic for backward compatibility
    if name == "double_integrator_box":
        from enerdynamics.envs.double_integrator_box import DoubleIntegratorBoxEnv
        return DoubleIntegratorBoxEnv(**kwargs)
    elif name == "double_integrator_box_2d":
        from enerdynamics.envs.double_integrator_box_2d import DoubleIntegratorBox2DEnv
        return DoubleIntegratorBox2DEnv(**kwargs)
    elif name == "single_integrator_box_2d":
        from enerdynamics.envs.single_integrator_box_2d import SingleIntegratorBox2DEnv
        return SingleIntegratorBox2DEnv(**kwargs)
    elif name == "drone_box_3d":
        from enerdynamics.envs.drone_box_3d import DroneBox3DEnv
        return DroneBox3DEnv(**kwargs)
    elif name == "drone_full_3d":
        from enerdynamics.envs.drone_full_3d import DroneFull3DEnv
        return DroneFull3DEnv(**kwargs)
    elif name == "drone_full_3d_physics":
        from enerdynamics.envs.drone_full_3d_physics import DroneFull3DPhysicsEnv
        return DroneFull3DPhysicsEnv(**kwargs)
    elif name == "drone_full_3d_mujoco":
        from enerdynamics.envs.drone_full_3d_mujoco import DroneFull3DMujocoEnv
        return DroneFull3DMujocoEnv(**kwargs)
    elif name == "drone_full_3d_isaac":
        from enerdynamics.envs.drone_full_3d_isaac import DroneFull3DIsaacEnv
        return DroneFull3DIsaacEnv(**kwargs)
    else:
        available = []
        try:
            from enerdynamics.core.registry.environments import get_environment_registry
            registry = get_environment_registry()
            available = registry.list_available()
        except (ImportError, AttributeError):
            pass
        if available:
            raise ValueError(
                f"Unknown environment name: {name}. "
                f"Available environments: {available}"
            )
        else:
            raise ValueError(f"Unknown environment name: {name}")


def make_env_adapter(
    env: Any,
    backend: str = "numpy",
    obstacles: Optional[list] = None,
    **kwargs
):
    """
    Factory function to create environment adapter.
    
    Automatically detects environment type and wraps with appropriate adapter.
    
    Args:
        env: Environment to wrap (Gymnasium, Brax, or native enerdynamics)
        backend: Computational backend ("numpy", "jax", "torch")
        obstacles: List of obstacles (optional)
        **kwargs: Additional adapter parameters
        
    Returns:
        UnifiedEnvAdapter instance
    """
    from enerdynamics.envs.adapters.unified_adapter import UnifiedEnvAdapter
    return UnifiedEnvAdapter(env, backend=backend, obstacles=obstacles, **kwargs)


def make_energy(env_name: str) -> LegacyEnergyFunctional:
    """
    Factory function to create energy functionals for environments using the registry system.
    
    Args:
        env_name: Environment name
        
    Returns:
        LegacyEnergyFunctional instance
        
    Raises:
        ValueError: If environment name is not recognized
    """
    # Try to use energy registry first
    try:
        from enerdynamics.core.registry.energy import get_energy_registry
        registry = get_energy_registry()
        # Check if registered (has factory or class)
        if registry.is_registered(env_name):
            # Use registry.create which handles factories automatically
            return registry.create(env_name)
    except (ImportError, AttributeError, TypeError):
        # Registry not available or error, fall back to legacy hardcoded logic
        pass
    except ValueError:
        # Not found in registry, fall back to legacy
        pass
    
    # Fallback to legacy hardcoded logic for backward compatibility
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
            "task": EnergyTerm(task_energy, 2.0),
            "box": EnergyTerm(box_energy, 1.0),
        })
    elif env_name == "single_integrator_box_2d":
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
    elif env_name == "drone_box_3d":
        def task_energy(x, u, ctx):
            pos = x[:3]
            vel = x[3:6]
            return jnp.sum((pos - jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)) ** 2) + jnp.sum(vel ** 2)

        def box_energy(x, u, ctx):
            p_max, v_max = 2.0, 2.0
            pos_violate = jnp.maximum(0.0, jnp.abs(x[:3]) - p_max)
            vel_violate = jnp.maximum(0.0, jnp.abs(x[3:6]) - v_max)
            pen = jnp.sum(pos_violate ** 2 + vel_violate ** 2)
            return pen

        return LegacyEnergyFunctional({
            "task": EnergyTerm(task_energy, 2.0),
            "box": EnergyTerm(box_energy, 1.0),
        })
    elif env_name == "drone_full_3d":
        def task_energy(x, u, ctx):
            pos = x[:3]
            vel = x[3:6]
            euler = x[6:9]
            target = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
            pos_err = jnp.sum((pos - target) ** 2)
            vel_err = jnp.sum(vel ** 2)
            orientation_err = jnp.sum(euler ** 2)
            return pos_err + 0.1 * vel_err + 0.1 * orientation_err

        def box_energy(x, u, ctx):
            p_max, v_max = 2.0, 2.0
            pos_violate = jnp.maximum(0.0, jnp.abs(x[:3]) - p_max)
            vel_violate = jnp.maximum(0.0, jnp.abs(x[3:6]) - v_max)
            # Orientation bounds
            euler_violate = jnp.maximum(0.0, jnp.abs(x[6:9]) - jnp.pi)
            pen = jnp.sum(pos_violate ** 2 + vel_violate ** 2 + euler_violate ** 2)
            return pen

        return LegacyEnergyFunctional({
            "task": EnergyTerm(task_energy, 2.0),
            "box": EnergyTerm(box_energy, 1.0),
        })
    elif env_name == "drone_full_3d_physics":
        # Same energy functional as drone_full_3d
        def task_energy(x, u, ctx):
            pos = x[:3]
            vel = x[3:6]
            euler = x[6:9]
            target = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
            pos_err = jnp.sum((pos - target) ** 2)
            vel_err = jnp.sum(vel ** 2)
            orientation_err = jnp.sum(euler ** 2)
            return pos_err + 0.1 * vel_err + 0.1 * orientation_err

        def box_energy(x, u, ctx):
            p_max, v_max = 2.0, 2.0
            pos_violate = jnp.maximum(0.0, jnp.abs(x[:3]) - p_max)
            vel_violate = jnp.maximum(0.0, jnp.abs(x[3:6]) - v_max)
            # Orientation bounds
            euler_violate = jnp.maximum(0.0, jnp.abs(x[6:9]) - jnp.pi)
            pen = jnp.sum(pos_violate ** 2 + vel_violate ** 2 + euler_violate ** 2)
            return pen

        return LegacyEnergyFunctional({
            "task": EnergyTerm(task_energy, 2.0),
            "box": EnergyTerm(box_energy, 1.0),
        })
    else:
        raise ValueError(f"Unknown environment name: {env_name}")

