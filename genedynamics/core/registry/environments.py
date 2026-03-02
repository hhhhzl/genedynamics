"""
Environment registry for environment implementations.

This registry manages environment class registrations. Environments can be
registered either programmatically or via entry points.
"""

from typing import Type, Callable, Optional, Any
from genedynamics.core.registry.base import BaseRegistry

# Type hint for environment classes (using Any for flexibility as environments
# may not all inherit from a common base class)
EnvType = Any

# Global environment registry instance
_environment_registry = BaseRegistry[EnvType]("genedynamics.envs")


def register_env(
    name: str,
    env_class: Type[EnvType],
    factory: Optional[Callable[..., EnvType]] = None
) -> None:
    """
    Register an environment implementation programmatically.
    
    Args:
        name: Environment name (e.g., "double_integrator_box")
        env_class: Environment class
        factory: Optional factory function
    
    Examples:
        >>> from genedynamics.core.registry.environments import register_env
        >>> register_env("my_env", MyEnvClass)
    """
    _environment_registry.register(name, env_class, factory)


def get_environment_registry() -> BaseRegistry[EnvType]:
    """
    Get the global environment registry.
    
    Returns:
        Environment registry instance
    """
    return _environment_registry

