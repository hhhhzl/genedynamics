"""
Unified registry system for multi-backend constraint components.

This module provides a centralized registry for all constraint system components:
- Convexifiers (CFS, CBF, etc.)
- Operators (QP filters, projections, etc.)
- Solvers (QP solvers, etc.)
- Schedulers (optional)

Registry structure:
    {module_type: {component_name: {backend: impl_class}}}

Example:
    registry["convexifier"]["cfs"]["jax"] = CFSJaxConvexifier
    registry["convexifier"]["cfs"]["numpy"] = CFSNumpyConvexifier
    registry["operator"]["per_step_qp"]["jax"] = PerStepQPJaxOperator

Performance optimizations:
- Fast lookup: O(1) dictionary access
- Lazy loading: Components loaded on demand
- Auto-selection: Automatically selects best available backend
"""

from typing import Dict, Type, Optional, List
from abc import ABC
import warnings


class UnifiedRegistry:
    """
    Unified registry system for multi-backend constraint components.
    
    This registry manages implementations across different backends (numpy, JAX, PyTorch, Rust)
    for all constraint system modules (convexifiers, operators, solvers, schedulers).
    
    Registry structure:
        {module_type: {component_name: {backend: impl_class}}}
    
    Example:
        >>> registry = UnifiedRegistry()
        >>> registry.register("convexifier", "cfs", "jax", CFSJaxConvexifier)
        >>> impl_class = registry.get("convexifier", "cfs", "jax")
    
    Performance:
        - O(1) lookup time
        - Minimal memory overhead
        - Supports lazy loading
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._registry: Dict[str, Dict[str, Dict[str, Type]]] = {}
    
    def register(
        self,
        module_type: str,      # "convexifier", "operator", "solver", "scheduler"
        component_name: str,    # "cfs", "cbf", "per_step_qp", "qpax"
        backend: str,           # "jax", "numpy", "torch", "rust"
        impl_class: Type
    ) -> None:
        """
        Register an implementation.
        
        Args:
            module_type: Module type ("convexifier", "operator", "solver", "scheduler")
            component_name: Component name (e.g., "cfs", "cbf", "per_step_qp")
            backend: Backend name ("jax", "numpy", "torch", "rust")
            impl_class: Implementation class
            
        Raises:
            ValueError: If arguments are invalid
        """
        if not isinstance(module_type, str) or not module_type:
            raise ValueError(f"module_type must be non-empty string, got: {module_type}")
        if not isinstance(component_name, str) or not component_name:
            raise ValueError(f"component_name must be non-empty string, got: {component_name}")
        if not isinstance(backend, str) or not backend:
            raise ValueError(f"backend must be non-empty string, got: {backend}")
        
        # Initialize nested dictionaries if needed
        if module_type not in self._registry:
            self._registry[module_type] = {}
        if component_name not in self._registry[module_type]:
            self._registry[module_type][component_name] = {}
        
        # Register implementation
        self._registry[module_type][component_name][backend] = impl_class
    
    def get(
        self,
        module_type: str,
        component_name: str,
        backend: str
    ) -> Optional[Type]:
        """
        Get implementation class.
        
        Args:
            module_type: Module type
            component_name: Component name
            backend: Backend name
            
        Returns:
            Implementation class, or None if not found
        """
        return (
            self._registry
            .get(module_type, {})
            .get(component_name, {})
            .get(backend)
        )
    
    def list_backends(
        self,
        module_type: str,
        component_name: str
    ) -> List[str]:
        """
        List all available backends for a component.
        
        Args:
            module_type: Module type
            component_name: Component name
            
        Returns:
            Sorted list of available backend names
        """
        backends = (
            self._registry
            .get(module_type, {})
            .get(component_name, {})
            .keys()
        )
        return sorted(backends)
    
    def list_components(
        self,
        module_type: str
    ) -> List[str]:
        """
        List all available components for a module type.
        
        Args:
            module_type: Module type
            
        Returns:
            Sorted list of component names
        """
        components = self._registry.get(module_type, {}).keys()
        return sorted(components)
    
    def list_module_types(self) -> List[str]:
        """
        List all registered module types.
        
        Returns:
            Sorted list of module type names
        """
        return sorted(self._registry.keys())
    
    def auto_select_backend(
        self,
        module_type: str,
        component_name: str,
        preferred: Optional[str] = None
    ) -> Optional[str]:
        """
        Automatically select best available backend.
        
        Selection priority:
        1. Preferred backend (if available)
        2. JAX (if available)
        3. NumPy (if available)
        4. PyTorch (if available)
        5. Rust (if available)
        6. First available backend
        
        Args:
            module_type: Module type
            component_name: Component name
            preferred: Preferred backend name (optional)
            
        Returns:
            Selected backend name, or None if no backends available
        """
        available = self.list_backends(module_type, component_name)
        if not available:
            return None
        
        # Check preferred backend first
        if preferred and preferred in available:
            return preferred
        
        # Priority order: jax > numpy > torch > rust > others
        priority_order = ["jax", "numpy", "torch", "rust"]
        for backend in priority_order:
            if backend in available:
                return backend
        
        # Return first available if no priority match
        return available[0]
    
    def is_registered(
        self,
        module_type: str,
        component_name: Optional[str] = None,
        backend: Optional[str] = None
    ) -> bool:
        """
        Check if component is registered.
        
        Args:
            module_type: Module type
            component_name: Component name (optional, checks all if None)
            backend: Backend name (optional, checks all if None)
            
        Returns:
            True if registered, False otherwise
        """
        if module_type not in self._registry:
            return False
        
        if component_name is None:
            return True
        
        if component_name not in self._registry[module_type]:
            return False
        
        if backend is None:
            return True
        
        return backend in self._registry[module_type][component_name]
    
    def unregister(
        self,
        module_type: str,
        component_name: str,
        backend: str
    ) -> bool:
        """
        Unregister an implementation.
        
        Args:
            module_type: Module type
            component_name: Component name
            backend: Backend name
            
        Returns:
            True if unregistered, False if not found
        """
        try:
            del self._registry[module_type][component_name][backend]
            return True
        except KeyError:
            return False
    
    def clear(self) -> None:
        """Clear all registrations."""
        self._registry.clear()
    
    def __repr__(self) -> str:
        """String representation."""
        total_components = sum(
            len(backends)
            for module in self._registry.values()
            for backends in module.values()
        )
        return f"UnifiedRegistry({total_components} components across {len(self._registry)} module types)"


# Global registry instance
_global_registry = UnifiedRegistry()


def get_registry() -> UnifiedRegistry:
    """
    Get the global registry instance.
    
    Returns:
        Global UnifiedRegistry instance
    """
    return _global_registry


def register(module_type: str, component_name: str, backend: str):
    """
    Decorator for registering implementations.
    
    Usage:
        >>> @register("convexifier", "cfs", "jax")
        ... class CFSJaxConvexifier:
        ...     pass
    
    Args:
        module_type: Module type
        component_name: Component name
        backend: Backend name
        
    Returns:
        Decorator function
    """
    def decorator(impl_class: Type) -> Type:
        _global_registry.register(module_type, component_name, backend, impl_class)
        return impl_class
    return decorator


