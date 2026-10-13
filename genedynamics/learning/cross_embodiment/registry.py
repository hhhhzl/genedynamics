"""Lazy providers built on the repository's canonical BaseRegistry."""
from genedynamics.registry_base import BaseRegistry
from .backend_impl import ExecutionBackend


def _jax_factory(**kwargs):
    from .backends.jax.execution import JaxExecutionBackend
    return JaxExecutionBackend(**kwargs)


def _numpy_factory(**kwargs):
    from genedynamics.envs.domains.manipulation.cross_embodiment.backends.mujoco import MujocoExecutionBackend
    return MujocoExecutionBackend(**kwargs)


backend_registry = BaseRegistry[ExecutionBackend]("genedynamics.cross_embodiment.backends")
backend_registry.register("jax", ExecutionBackend, factory=_jax_factory)
backend_registry.register("numpy", ExecutionBackend, factory=_numpy_factory)
