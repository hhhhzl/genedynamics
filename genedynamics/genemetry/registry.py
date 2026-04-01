"""
Multi-backend component registry for genemetry.

Maintains its own registry instance, independent of the constraint-system
registry in ``genedynamics.core.constraints.core.registry``.

Structure
---------
::

    {module_type: {component_name: {backend: impl_class}}}

Module types : ``"manifold"``, ``"ops"``, ``"retraction"``, ``"gate"``,
               ``"pipeline"``, ``"step"``, ``"window"``
Backends     : ``"jax"``, ``"numpy"``, ``"torch"``, ``"rust"``
"""

from typing import Dict, Type, Optional, List


class GenemetryRegistry:
    """Multi-backend component registry for genemetry.

    Mirrors the interface of
    ``genedynamics.core.constraints.core.registry.UnifiedRegistry``
    but lives in a separate global instance so that geometry components
    and constraint components are decoupled.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Dict[str, Dict[str, Type]]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        module_type: str,
        component_name: str,
        backend: str,
        impl_class: Type,
    ) -> None:
        """Register a backend implementation class."""
        if module_type not in self._registry:
            self._registry[module_type] = {}
        if component_name not in self._registry[module_type]:
            self._registry[module_type][component_name] = {}
        self._registry[module_type][component_name][backend] = impl_class

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(
        self,
        module_type: str,
        component_name: str,
        backend: str,
    ) -> Optional[Type]:
        """Return the implementation class, or ``None`` if not found."""
        return (
            self._registry
            .get(module_type, {})
            .get(component_name, {})
            .get(backend)
        )

    def list_backends(
        self, module_type: str, component_name: str
    ) -> List[str]:
        """Sorted list of available backends for a component."""
        return sorted(
            self._registry
            .get(module_type, {})
            .get(component_name, {})
            .keys()
        )

    def list_components(self, module_type: str) -> List[str]:
        """Sorted list of registered component names under *module_type*."""
        return sorted(self._registry.get(module_type, {}).keys())

    def list_module_types(self) -> List[str]:
        """Sorted list of all registered module types."""
        return sorted(self._registry.keys())

    def auto_select_backend(
        self,
        module_type: str,
        component_name: str,
        preferred: Optional[str] = None,
    ) -> Optional[str]:
        """Pick the best available backend.

        Priority: *preferred* > jax > numpy > torch > rust > first available.
        """
        available = self.list_backends(module_type, component_name)
        if not available:
            return None
        if preferred and preferred in available:
            return preferred
        for b in ("jax", "numpy", "torch", "rust"):
            if b in available:
                return b
        return available[0]

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def unregister(
        self, module_type: str, component_name: str, backend: str
    ) -> bool:
        """Remove a registration.  Returns ``True`` if found."""
        try:
            del self._registry[module_type][component_name][backend]
            return True
        except KeyError:
            return False

    def clear(self) -> None:
        """Remove all registrations."""
        self._registry.clear()

    def __repr__(self) -> str:
        total = sum(
            len(backends)
            for module in self._registry.values()
            for backends in module.values()
        )
        return (
            f"GenemetryRegistry({total} implementations "
            f"across {len(self._registry)} module types)"
        )


# ======================================================================
# Global singleton
# ======================================================================

_genemetry_registry = GenemetryRegistry()


def get_genemetry_registry() -> GenemetryRegistry:
    """Return the global genemetry registry instance."""
    return _genemetry_registry


def register_genemetry(
    module_type: str, component_name: str, backend: str
):
    """Class decorator that registers a backend implementation.

    Usage::

        @register_genemetry("manifold", "sdf", "jax")
        class SdfManifoldJax(ConstraintManifold):
            ...
    """

    def decorator(impl_class: Type) -> Type:
        _genemetry_registry.register(
            module_type, component_name, backend, impl_class
        )
        return impl_class

    return decorator
