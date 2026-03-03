"""
EDOC backend registry for EDOC backend implementations.

This registry manages EDOC backend implementations (JAX, NumPy, etc.).
Similar to ProjectionTypeRegistry but specifically for EDOC backends.
"""

from typing import Dict, Type, Optional
from importlib.metadata import entry_points
import warnings

from genedynamics.core.registry.base import BaseRegistry


class EDOCBackendRegistry:
    """
    Registry for EDOC backend implementations.
    
    This registry supports multiple backend implementations for EDOC.
    Entry point naming convention: "edoc.backend_name" = "module:Class"
    
    Examples:
        >>> registry = EDOCBackendRegistry()
        >>> registry.register("jax", EDOCBackendJax)
        >>> registry.register("numpy", EDOCBackendNumpy)
        >>> impl_class = registry.get("jax")
        >>> backends = registry.list_backends()
    """
    
    def __init__(self):
        """Initialize EDOC backend registry."""
        self._entry_points: Dict[str, Type] = {}
        self._programmatic: Dict[str, Type] = {}
        self._loaded = False
    
    def _load_entry_points(self) -> None:
        """Lazy-load entry points."""
        if self._loaded:
            return
        
        try:
            eps = entry_points(group="genedynamics.solvers.edoc_backends")
        except TypeError:
            eps = entry_points().get("genedynamics.solvers.edoc_backends", [])
        
        for ep in eps:
            try:
                # Entry point name format: "backend_name"
                # Examples: "jax", "numpy"
                backend_name = ep.name
                impl_class = ep.load()
                self._entry_points[backend_name] = impl_class
            except (ImportError, AttributeError, ModuleNotFoundError) as e:
                warnings.warn(
                    f"Failed to load EDOC backend entry point '{ep.name}': {e}",
                    UserWarning,
                    stacklevel=2
                )
        
        self._loaded = True
    
    def register(
        self,
        backend_name: str,
        impl_class: Type
    ) -> None:
        """
        Register an EDOC backend implementation.
        
        Args:
            backend_name: Backend name (e.g., "jax", "numpy")
            impl_class: Implementation class
        
        Examples:
            >>> registry.register("jax", EDOCBackendJax)
        """
        if not isinstance(backend_name, str) or not backend_name:
            raise ValueError(f"Backend name must be a non-empty string, got: {backend_name}")
        
        self._programmatic[backend_name] = impl_class
    
    def get(
        self,
        backend_name: str
    ) -> Optional[Type]:
        """
        Get implementation class for backend.
        
        Args:
            backend_name: Backend name
        
        Returns:
            Implementation class, or None if not found
        """
        self._load_entry_points()
        
        # Priority: programmatic > entry points
        if backend_name in self._programmatic:
            return self._programmatic[backend_name]
        
        if backend_name in self._entry_points:
            return self._entry_points[backend_name]
        
        return None
    
    def list_backends(self) -> list[str]:
        """
        List all available backends.
        
        Returns:
            Sorted list of backend names
        """
        self._load_entry_points()
        backends = set(self._programmatic.keys())
        backends.update(self._entry_points.keys())
        return sorted(backends)
    
    def is_registered(self, backend_name: str) -> bool:
        """
        Check if a backend is registered.
        
        Args:
            backend_name: Backend name
        
        Returns:
            True if registered, False otherwise
        """
        self._load_entry_points()
        return (backend_name in self._programmatic or
                backend_name in self._entry_points)


# Global EDOC backend registry instance
_edoc_backend_registry = EDOCBackendRegistry()


def register_edoc_backend(backend_name: str):
    """
    Decorator for registering EDOC backend implementations.
    
    Args:
        backend_name: Backend name
    
    Returns:
        Decorator function
    
    Examples:
        >>> @register_edoc_backend("jax")
        ... class EDOCBackendJax:
        ...     pass
    """
    def decorator(impl_class: Type) -> Type:
        _edoc_backend_registry.register(backend_name, impl_class)
        return impl_class
    return decorator


def get_edoc_backend_registry() -> EDOCBackendRegistry:
    """
    Get the global EDOC backend registry.
    
    Returns:
        EDOC backend registry instance
    """
    return _edoc_backend_registry

