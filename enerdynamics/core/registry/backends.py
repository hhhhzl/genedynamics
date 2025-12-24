"""
Backend registry for computational backend implementations.

This registry manages backend implementations (JAX, NumPy, PyTorch, etc.)
and allows automatic selection based on runtime context.
"""

from typing import Type
from enerdynamics.core.backends import Backend
from enerdynamics.core.registry.base import BaseRegistry

# Global backend registry instance
_backend_registry = BaseRegistry[Backend]("enerdynamics.backends")


def register_backend(name: str, backend_class: Type[Backend], factory=None):
    """
    Register a backend implementation programmatically.
    
    Args:
        name: Backend name (e.g., "numpy", "jax", "torch")
        backend_class: Backend implementation class
        factory: Optional factory function
    
    Examples:
        >>> from enerdynamics.core.registry.backends import register_backend
        >>> register_backend("my_backend", MyBackendClass)
    """
    _backend_registry.register(name, backend_class, factory)


def get_backend_registry() -> BaseRegistry[Backend]:
    """
    Get the global backend registry.
    
    Returns:
        Backend registry instance
    """
    return _backend_registry

