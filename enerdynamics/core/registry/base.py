"""
Base registry implementation with entry points support.

This module provides a generic registry base class that supports both
entry points (for installed packages) and programmatic registration
(for user code). All component registries inherit from this base class.
"""

from typing import Dict, Type, Optional, Callable, Any, Generic, TypeVar
from importlib.metadata import entry_points
import warnings

T = TypeVar('T')


class BaseRegistry(Generic[T]):
    """
    Generic registry base class with entry points support.
    
    Supports both entry points (for installed packages) and programmatic
    registration (for user code). Entry points are loaded lazily on first access
    to improve startup performance.
    
    Type Parameters:
        T: The type of objects registered in this registry.
    
    Examples:
        >>> registry = BaseRegistry("my.plugin.group")
        >>> registry.register("my_plugin", MyPluginClass)
        >>> instance = registry.create("my_plugin", arg1=1, arg2=2)
    """
    
    def __init__(self, entry_point_group: str):
        """
        Initialize registry.
        
        Args:
            entry_point_group: Entry point group name (e.g., "enerdynamics.backends")
        """
        self.entry_point_group = entry_point_group
        self._entry_points: Dict[str, Type[T]] = {}
        self._programmatic: Dict[str, Type[T]] = {}
        self._factories: Dict[str, Callable[..., T]] = {}
        self._loaded = False
    
    def _load_entry_points(self) -> None:
        """
        Lazy-load entry points from installed packages.
        
        This is called automatically on first access. Entry points are only
        loaded once and cached for performance.
        """
        if self._loaded:
            return
        
        try:
            # Python 3.10+ way (preferred)
            eps = entry_points(group=self.entry_point_group)
        except TypeError:
            # Python 3.8/3.9 compatibility
            eps = entry_points().get(self.entry_point_group, [])
        
        for ep in eps:
            try:
                impl_class = ep.load()
                self._entry_points[ep.name] = impl_class
            except (ImportError, AttributeError, ModuleNotFoundError) as e:
                # Skip if dependencies are missing (e.g., JAX not installed)
                # Only warn, don't fail, as optional dependencies are expected
                warnings.warn(
                    f"Failed to load entry point '{ep.name}' from group "
                    f"'{self.entry_point_group}': {e}. "
                    f"Install required dependencies to enable this component.",
                    UserWarning,
                    stacklevel=2
                )
        
        self._loaded = True
    
    def register(
        self,
        name: str,
        impl_class: Type[T],
        factory: Optional[Callable[..., T]] = None
    ) -> None:
        """
        Programmatically register an implementation.
        
        Programmatic registrations take precedence over entry points.
        This allows users to override entry point registrations or register
        custom implementations without creating packages.
        
        Args:
            name: Registration name (must be unique)
            impl_class: Implementation class
            factory: Optional factory function that takes **kwargs and returns
                    an instance. If provided, factory is used instead of
                    direct instantiation.
        
        Examples:
            >>> registry.register("my_backend", MyBackendClass)
            >>> registry.register("my_backend", MyBackendClass, factory=lambda **kw: MyBackendClass(**kw))
        """
        if not isinstance(name, str) or not name:
            raise ValueError(f"Registration name must be a non-empty string, got: {name}")
        
        self._programmatic[name] = impl_class
        if factory is not None:
            self._factories[name] = factory
    
    def get_class(self, name: str) -> Optional[Type[T]]:
        """
        Get implementation class by name (without instantiating).
        
        Args:
            name: Registration name
            
        Returns:
            Implementation class, or None if not found
        """
        self._load_entry_points()
        
        # Priority: programmatic > entry points
        return self._programmatic.get(name) or self._entry_points.get(name)
    
    def create(self, name: str, **kwargs: Any) -> T:
        """
        Create an instance of the registered implementation.
        
        Args:
            name: Registration name
            **kwargs: Arguments passed to constructor or factory function
            
        Returns:
            Instance of the registered class
            
        Raises:
            ValueError: If name is not found in registry
            TypeError: If instantiation fails (wrong arguments, etc.)
        """
        self._load_entry_points()
        
        # Priority: factory > programmatic > entry points
        if name in self._factories:
            return self._factories[name](**kwargs)
        
        impl_class = self._programmatic.get(name) or self._entry_points.get(name)
        
        if impl_class is None:
            available = self.list_available()
            raise ValueError(
                f"'{name}' not found in registry '{self.entry_point_group}'. "
                f"Available: {available}"
            )
        
        try:
            return impl_class(**kwargs)
        except Exception as e:
            raise TypeError(
                f"Failed to instantiate '{name}' from registry '{self.entry_point_group}': {e}"
            ) from e
    
    def list_available(self) -> list[str]:
        """
        List all available registration names.
        
        Returns:
            Sorted list of all available names (programmatic + entry points)
        """
        self._load_entry_points()
        all_names = set(self._programmatic.keys())
        all_names.update(self._entry_points.keys())
        return sorted(all_names)
    
    def is_registered(self, name: str) -> bool:
        """
        Check if a name is registered (either programmatically or via entry points).
        
        Args:
            name: Registration name to check
            
        Returns:
            True if name is registered, False otherwise
        """
        self._load_entry_points()
        return name in self._programmatic or name in self._entry_points
    
    def clear(self) -> None:
        """
        Clear all programmatic registrations.
        
        Note: Entry points cannot be cleared (they are reloaded on next access).
        This is mainly useful for testing.
        """
        self._programmatic.clear()
        self._factories.clear()

