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


def register_environment(name: str, *aliases: str):
    """
    Class decorator: register an environment class under ``name`` (+ aliases).

    Co-locates registration with the class definition so the domain-subpackage
    layout dispatches via the registry instead of a make_env if/elif chain.

    Examples:
        >>> @register_environment("quadruped_go2_mjx")
        ... class QuadrupedGo2MjxEnv: ...
        >>> @register_environment("quadruped_go2_mjx_trot", "quadruped_go2_dial")
        ... class QuadrupedGo2TrotMjxEnv(QuadrupedGo2MjxEnv): ...
    """
    def deco(cls):
        _environment_registry.register(name, cls)
        for a in aliases:
            _environment_registry.register(a, cls)
        return cls
    return deco


def register_environment_factory(
    name: str,
    factory: Callable[..., EnvType],
    *aliases: str,
) -> None:
    """
    Register an environment produced by a factory function (e.g. brax helpers
    like ``make_brax_go2``) under ``name`` (+ aliases). ``BaseRegistry.create``
    already prefers the factory over the placeholder class.
    """
    _environment_registry.register(name, object, factory=factory)
    for a in aliases:
        _environment_registry.register(a, object, factory=factory)


def register_unavailable_environment(name: str, exc: Exception) -> None:
    """Register a placeholder for an env whose optional backend failed to import.

    The placeholder raises ``ImportError`` when instantiated (``make_env``), so
    callers that ``try: make_env(...) except ImportError: skip`` keep skipping
    when an optional backend (mujoco / isaac / brax / mjx) is absent — preserving
    the behaviour of the former make_env if/elif (which surfaced the ImportError).
    """
    msg = f"Environment '{name}' is unavailable (optional backend failed to import): {exc}"

    def _factory(**_kwargs):
        raise ImportError(msg)

    _environment_registry.register(name, object, factory=_factory)


def get_environment_registry() -> BaseRegistry[EnvType]:
    """
    Get the global environment registry.

    Returns:
        Environment registry instance
    """
    return _environment_registry

