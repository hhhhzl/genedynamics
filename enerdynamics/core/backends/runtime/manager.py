"""
Runtime Backend Manager for managing computational backends.

This module provides a global backend manager that allows switching
between different backends (JAX, NumPy, PyTorch, etc.) and devices (CPU, GPU, WebGPU).
"""

from typing import Optional, Dict, Any
from enerdynamics.core.backends.runtime.jax_backend import JaxBackend
from enerdynamics.core.backends.runtime.numpy_backend import NumpyBackend
from enerdynamics.core.backends.runtime.torch_backend import TorchBackend


class RuntimeBackendManager:
    """
    Global runtime backend manager.
    
    This is a singleton class that manages the global computational backend.
    It allows switching between different backends and devices at runtime.
    
    Usage:
        # Set backend at startup
        RuntimeBackendManager.set_backend("jax", device="gpu")
        
        # Use in code
        backend = RuntimeBackendManager.get_backend()
        rng = backend.create_rng(42)
        
        # Or get backend directly
        backend = RuntimeBackendManager.get_backend("numpy", device="cpu")
    """
    
    _instance: Optional['RuntimeBackendManager'] = None
    _backend: Optional[Any] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def set_backend(cls, name: str = "jax", device: str = "cpu", **kwargs) -> None:
        """
        Set the global backend.
        
        Args:
            name: Backend name ("jax", "numpy", "torch", etc.)
            device: Device name ("cpu", "gpu", "webgpu")
            **kwargs: Additional backend-specific arguments
            
        Raises:
            ValueError: If backend name is not recognized
            ImportError: If required backend library is not installed
        """
        backend_map = {
            "jax": JaxBackend,
            "numpy": NumpyBackend,
            "torch": TorchBackend,
        }
        
        if name not in backend_map:
            raise ValueError(
                f"Unknown backend: {name}. "
                f"Available backends: {list(backend_map.keys())}"
            )
        
        backend_class = backend_map[name]
        cls._backend = backend_class(device=device, **kwargs)
    
    @classmethod
    def get_backend(cls, name: Optional[str] = None, device: Optional[str] = None, **kwargs) -> Any:
        """
        Get the current backend or create a new one.
        
        If name is provided, creates a new backend instance (doesn't change global).
        If name is None, returns the global backend (creates default if not set).
        
        Args:
            name: Optional backend name. If None, returns global backend.
            device: Optional device name. If None, uses "cpu".
            **kwargs: Additional backend-specific arguments
            
        Returns:
            Backend instance
            
        Raises:
            ValueError: If backend name is not recognized
            ImportError: If required backend library is not installed
        """
        if name is not None:
            # Create a new backend instance (doesn't affect global)
            backend_map = {
                "jax": JaxBackend,
                "numpy": NumpyBackend,
                "torch": TorchBackend,
            }
            
            if name not in backend_map:
                raise ValueError(
                    f"Unknown backend: {name}. "
                    f"Available backends: {list(backend_map.keys())}"
                )
            
            backend_class = backend_map[name]
            if device is None:
                device = "cpu"
            return backend_class(device=device, **kwargs)
        
        # Return global backend
        if cls._backend is None:
            # Default to JAX if not set
            cls.set_backend("jax", device="cpu" if device is None else device, **kwargs)
        
        return cls._backend
    
    @classmethod
    def reset(cls) -> None:
        """
        Reset to default backend (JAX on CPU).
        
        This clears the global backend, causing it to be recreated
        with default settings on next access.
        """
        cls._backend = None
    
    @classmethod
    def get_backend_name(cls) -> str:
        """
        Get the name of the current global backend.
        
        Returns:
            Backend name (e.g., "jax", "numpy", "torch")
        """
        backend = cls.get_backend()
        return backend.name
    
    @classmethod
    def get_device(cls) -> str:
        """
        Get the device of the current global backend.
        
        Returns:
            Device name (e.g., "cpu", "gpu")
        """
        backend = cls.get_backend()
        return backend.get_device()
    
    @classmethod
    def is_available(cls, name: str) -> bool:
        """
        Check if a backend is available (library installed).
        
        Args:
            name: Backend name to check
            
        Returns:
            True if backend is available, False otherwise
        """
        if name == "jax":
            try:
                import jax
                return True
            except ImportError:
                return False
        elif name == "numpy":
            return True  # NumPy is always available
        elif name == "torch":
            try:
                import torch
                return True
            except ImportError:
                return False
        else:
            return False

