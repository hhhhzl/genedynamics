"""
Action filter registry for action-space constraint implementations.

This registry manages action filter implementations with a two-level structure:
- Filter type (e.g., "cbf_double_integrator_2d")
- Backend (e.g., "numpy", "jax", "torch")

Structure: {filter_type: {backend_name: impl_class}}
"""

from typing import Dict, Type, Optional
from importlib.metadata import entry_points
import warnings


class ActionFilterRegistry:
    """
    Registry for action filter implementations with backend support.
    
    Entry point naming convention: "filter_type.backend_name" = "module:Class"
    
    Examples:
        >>> registry = ActionFilterRegistry()
        >>> registry.register("cbf_double_integrator_2d", "numpy", CBFActionFilterNumpy)
        >>> impl_class = registry.get("cbf_double_integrator_2d", "numpy")
    """
    
    def __init__(self):
        """Initialize action filter registry."""
        self._entry_points: Dict[str, Dict[str, Type]] = {}
        self._programmatic: Dict[str, Dict[str, Type]] = {}
        self._loaded = False
    
    def _load_entry_points(self) -> None:
        """Lazy-load entry points with two-level structure."""
        if self._loaded:
            return
        
        try:
            eps = entry_points(group="enerdynamics.constraints.action_filters")
        except TypeError:
            eps = entry_points().get("enerdynamics.constraints.action_filters", [])
        
        for ep in eps:
            try:
                # Entry point name format: "filter_type.backend_name"
                parts = ep.name.split(".", 1)
                if len(parts) == 2:
                    filter_type, backend_name = parts
                    impl_class = ep.load()
                    
                    if filter_type not in self._entry_points:
                        self._entry_points[filter_type] = {}
                    self._entry_points[filter_type][backend_name] = impl_class
                else:
                    warnings.warn(
                        f"Invalid entry point name format '{ep.name}' in group "
                        f"'enerdynamics.constraints.action_filters'. "
                        f"Expected format: 'filter_type.backend_name'",
                        UserWarning,
                        stacklevel=2
                    )
            except (ImportError, AttributeError, ModuleNotFoundError) as e:
                warnings.warn(
                    f"Failed to load action filter entry point '{ep.name}': {e}",
                    UserWarning,
                    stacklevel=2
                )
        
        self._loaded = True
    
    def register(
        self,
        filter_type: str,
        backend_name: str,
        impl_class: Type
    ) -> None:
        """
        Register an action filter implementation.
        
        Args:
            filter_type: Filter type name (e.g., "cbf_double_integrator_2d")
            backend_name: Backend name (e.g., "numpy", "jax", "torch")
            impl_class: Implementation class
        """
        if not isinstance(filter_type, str) or not filter_type:
            raise ValueError(f"Filter type must be a non-empty string, got: {filter_type}")
        if not isinstance(backend_name, str) or not backend_name:
            raise ValueError(f"Backend name must be a non-empty string, got: {backend_name}")
        
        if filter_type not in self._programmatic:
            self._programmatic[filter_type] = {}
        self._programmatic[filter_type][backend_name] = impl_class
    
    def get(
        self,
        filter_type: str,
        backend_name: str
    ) -> Optional[Type]:
        """
        Get implementation class for filter type and backend.
        
        Args:
            filter_type: Filter type name
            backend_name: Backend name
            
        Returns:
            Implementation class, or None if not found
        """
        self._load_entry_points()
        
        # Priority: programmatic > entry points
        if (filter_type in self._programmatic and
            backend_name in self._programmatic[filter_type]):
            return self._programmatic[filter_type][backend_name]
        
        if (filter_type in self._entry_points and
            backend_name in self._entry_points[filter_type]):
            return self._entry_points[filter_type][backend_name]
        
        return None
    
    def list_types(self) -> list[str]:
        """
        List all available filter types.
        
        Returns:
            Sorted list of filter type names
        """
        self._load_entry_points()
        types = set(self._programmatic.keys())
        types.update(self._entry_points.keys())
        return sorted(types)
    
    def list_backends(self, filter_type: str) -> list[str]:
        """
        List all available backends for a filter type.
        
        Args:
            filter_type: Filter type name
            
        Returns:
            Sorted list of backend names for this filter type
        """
        self._load_entry_points()
        backends = set()
        
        if filter_type in self._programmatic:
            backends.update(self._programmatic[filter_type].keys())
        if filter_type in self._entry_points:
            backends.update(self._entry_points[filter_type].keys())
        
        return sorted(backends)
    
    def is_registered(self, filter_type: str, backend_name: Optional[str] = None) -> bool:
        """
        Check if a filter type (and optionally backend) is registered.
        
        Args:
            filter_type: Filter type name
            backend_name: Optional backend name. If None, checks if type exists.
            
        Returns:
            True if registered, False otherwise
        """
        self._load_entry_points()
        
        if backend_name is None:
            return (filter_type in self._programmatic or
                   filter_type in self._entry_points)
        
        if (filter_type in self._programmatic and
            backend_name in self._programmatic[filter_type]):
            return True
        
        if (filter_type in self._entry_points and
            backend_name in self._entry_points[filter_type]):
            return True
        
        return False


# Global action filter registry instance
_action_filter_registry = ActionFilterRegistry()


def register_action_filter(filter_type: str, backend_name: str):
    """
    Decorator for registering action filter implementations.
    
    Args:
        filter_type: Filter type name
        backend_name: Backend name
    
    Returns:
        Decorator function
    
    Examples:
        >>> @register_action_filter("cbf_double_integrator_2d", "numpy")
        ... class CBFActionFilterNumpy:
        ...     pass
    """
    def decorator(impl_class: Type) -> Type:
        _action_filter_registry.register(filter_type, backend_name, impl_class)
        return impl_class
    return decorator


def get_action_filter_registry() -> ActionFilterRegistry:
    """
    Get the global action filter registry.
    
    Returns:
        Action filter registry instance
    """
    return _action_filter_registry

