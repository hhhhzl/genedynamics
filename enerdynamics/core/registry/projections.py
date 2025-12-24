"""
Projection registry supporting multiple projection types with backend implementations.

This registry manages projection implementations with a two-level structure:
- Projection type (e.g., "cfs", "gradient_descent", "learned")
- Backend (e.g., "numpy", "jax", "torch")

Structure: {projection_type: {backend_name: impl_class}}

Examples:
    - "cfs": {"numpy": CFSProjectionNumpy, "jax": CFSProjectionJax}
    - "gradient_descent": {"numpy": GDProjectionNumpy, "jax": GDProjectionJax}
"""

from typing import Dict, Type, Optional
from importlib.metadata import entry_points
import warnings

from enerdynamics.core.registry.base import BaseRegistry


class ProjectionTypeRegistry:
    """
    Registry for projection type implementations with backend support.
    
    This registry supports multiple projection types, each with multiple
    backend implementations. Entry point naming convention:
    "projection_type.backend_name" = "module:Class"
    
    Examples:
        >>> registry = ProjectionTypeRegistry()
        >>> registry.register("cfs", "numpy", CFSProjectionNumpy)
        >>> impl_class = registry.get("cfs", "numpy")
        >>> backends = registry.list_backends("cfs")
    """
    
    def __init__(self):
        """Initialize projection type registry."""
        self._entry_points: Dict[str, Dict[str, Type]] = {}
        self._programmatic: Dict[str, Dict[str, Type]] = {}
        self._loaded = False
    
    def _load_entry_points(self) -> None:
        """Lazy-load entry points with two-level structure."""
        if self._loaded:
            return
        
        try:
            eps = entry_points(group="enerdynamics.constraints.projections")
        except TypeError:
            eps = entry_points().get("enerdynamics.constraints.projections", [])
        
        for ep in eps:
            try:
                # Entry point name format: "projection_type.backend_name"
                # Examples: "cfs.numpy", "cfs.jax", "gradient_descent.numpy"
                parts = ep.name.split(".", 1)
                if len(parts) == 2:
                    projection_type, backend_name = parts
                    impl_class = ep.load()
                    
                    if projection_type not in self._entry_points:
                        self._entry_points[projection_type] = {}
                    self._entry_points[projection_type][backend_name] = impl_class
                else:
                    warnings.warn(
                        f"Invalid entry point name format '{ep.name}' in group "
                        f"'enerdynamics.constraints.projections'. "
                        f"Expected format: 'projection_type.backend_name'",
                        UserWarning,
                        stacklevel=2
                    )
            except (ImportError, AttributeError, ModuleNotFoundError) as e:
                warnings.warn(
                    f"Failed to load projection entry point '{ep.name}': {e}",
                    UserWarning,
                    stacklevel=2
                )
        
        self._loaded = True
    
    def register(
        self,
        projection_type: str,
        backend_name: str,
        impl_class: Type
    ) -> None:
        """
        Register a projection implementation.
        
        Args:
            projection_type: Projection type name (e.g., "cfs", "gradient_descent")
            backend_name: Backend name (e.g., "numpy", "jax", "torch")
            impl_class: Implementation class
        
        Examples:
            >>> registry.register("cfs", "numpy", CFSProjectionNumpy)
            >>> registry.register("cfs", "jax", CFSProjectionJax)
        """
        if not isinstance(projection_type, str) or not projection_type:
            raise ValueError(f"Projection type must be a non-empty string, got: {projection_type}")
        if not isinstance(backend_name, str) or not backend_name:
            raise ValueError(f"Backend name must be a non-empty string, got: {backend_name}")
        
        if projection_type not in self._programmatic:
            self._programmatic[projection_type] = {}
        self._programmatic[projection_type][backend_name] = impl_class
    
    def get(
        self,
        projection_type: str,
        backend_name: str
    ) -> Optional[Type]:
        """
        Get implementation class for projection type and backend.
        
        Args:
            projection_type: Projection type name
            backend_name: Backend name
            
        Returns:
            Implementation class, or None if not found
        
        Examples:
            >>> impl_class = registry.get("cfs", "numpy")
        """
        self._load_entry_points()
        
        # Priority: programmatic > entry points
        if (projection_type in self._programmatic and
            backend_name in self._programmatic[projection_type]):
            return self._programmatic[projection_type][backend_name]
        
        if (projection_type in self._entry_points and
            backend_name in self._entry_points[projection_type]):
            return self._entry_points[projection_type][backend_name]
        
        return None
    
    def list_types(self) -> list[str]:
        """
        List all available projection types.
        
        Returns:
            Sorted list of projection type names
        """
        self._load_entry_points()
        types = set(self._programmatic.keys())
        types.update(self._entry_points.keys())
        return sorted(types)
    
    def list_backends(self, projection_type: str) -> list[str]:
        """
        List all available backends for a projection type.
        
        Args:
            projection_type: Projection type name
            
        Returns:
            Sorted list of backend names for this projection type
        """
        self._load_entry_points()
        backends = set()
        
        if projection_type in self._programmatic:
            backends.update(self._programmatic[projection_type].keys())
        if projection_type in self._entry_points:
            backends.update(self._entry_points[projection_type].keys())
        
        return sorted(backends)
    
    def is_registered(self, projection_type: str, backend_name: Optional[str] = None) -> bool:
        """
        Check if a projection type (and optionally backend) is registered.
        
        Args:
            projection_type: Projection type name
            backend_name: Optional backend name. If None, checks if type exists.
            
        Returns:
            True if registered, False otherwise
        """
        self._load_entry_points()
        
        if backend_name is None:
            return (projection_type in self._programmatic or
                   projection_type in self._entry_points)
        
        if (projection_type in self._programmatic and
            backend_name in self._programmatic[projection_type]):
            return True
        
        if (projection_type in self._entry_points and
            backend_name in self._entry_points[projection_type]):
            return True
        
        return False


# Global projection registry instance
_projection_registry = ProjectionTypeRegistry()


def register_projection(projection_type: str, backend_name: str):
    """
    Decorator for registering projection implementations.
    
    Args:
        projection_type: Projection type name
        backend_name: Backend name
    
    Returns:
        Decorator function
    
    Examples:
        >>> @register_projection("cfs", "numpy")
        ... class CFSProjectionNumpy:
        ...     pass
    """
    def decorator(impl_class: Type) -> Type:
        _projection_registry.register(projection_type, backend_name, impl_class)
        return impl_class
    return decorator


def get_projection_registry() -> ProjectionTypeRegistry:
    """
    Get the global projection registry.
    
    Returns:
        Projection type registry instance
    """
    return _projection_registry


# Import backend implementations to trigger registration when registry is imported
try:
    from enerdynamics.core.constraints.projections import backends  # noqa: F401
except ImportError:
    pass  # Backends may not be available

