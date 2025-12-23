"""
Backend abstraction layer for pluggable compute backends.

This module provides a unified interface for different computational backends:
- JAX: for JIT compilation and automatic differentiation
- PyTorch: for neural network-based dynamics and energy models
- Rust/C++: for high-performance physics kernels (via FFI)
- WASM/Remote: for serverless or web-based execution

The Backend protocol defines the minimal set of operations needed by solvers,
allowing algorithms to be backend-agnostic.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class Backend(Protocol):
    """
    Protocol for computational backends.
    
    A backend provides tensor operations, parallelization, and JIT compilation
    capabilities. All solvers depend only on this interface, not specific frameworks.
    """
    
    name: str  # "jax", "torch", "numpy", "rust", "wasm", "remote"
    device: str  # "cpu", "gpu", "webgpu"
    
    @abstractmethod
    def tensor(self, x: Any, dtype: Optional[Any] = None) -> Any:
        """
        Convert input to backend tensor/array.
        
        Args:
            x: Input data (numpy array, list, etc.)
            dtype: Optional data type
            
        Returns:
            Backend tensor/array
        """
        ...
    
    @abstractmethod
    def concat(self, xs: list, axis: int = 0) -> Any:
        """
        Concatenate tensors along specified axis.
        
        Args:
            xs: List of tensors to concatenate
            axis: Axis along which to concatenate
            
        Returns:
            Concatenated tensor
        """
        ...
    
    @abstractmethod
    def randn(self, shape: tuple, rng_key: Optional[Any] = None) -> Any:
        """
        Generate random normal samples.
        
        Args:
            shape: Shape of output tensor
            rng_key: Optional random key/seed
            
        Returns:
            Random tensor
        """
        ...
    
    @abstractmethod
    def vmap(self, fn: Callable) -> Callable:
        """
        Vectorize a function over batch dimension.
        
        Args:
            fn: Function to vectorize
            
        Returns:
            Vectorized function
        """
        ...
    
    @abstractmethod
    def jit(self, fn: Callable) -> Callable:
        """
        JIT compile a function for performance.
        
        Args:
            fn: Function to compile
            
        Returns:
            Compiled function
        """
        ...
    
    @abstractmethod
    def create_rng(self, seed: int) -> Any:
        """
        Create a random number generator/PRNG key from seed.
        
        Args:
            seed: Random seed
            
        Returns:
            RNG state or PRNG key (backend-specific)
        """
        ...
    
    @abstractmethod
    def split_rng(self, rng: Any, num: int = 2) -> Any:
        """
        Split RNG state/key into multiple states/keys.
        
        For JAX, this splits PRNG keys. For NumPy/PyTorch, this may
        return the same state or create new generators.
        
        Args:
            rng: RNG state or PRNG key
            num: Number of states/keys to create
            
        Returns:
            Tuple of RNG states/keys or single state/key
        """
        ...
    
    @abstractmethod
    def uniform(self, rng: Any, shape: tuple, minval: float = 0.0, maxval: float = 1.0) -> Any:
        """
        Generate uniform random samples.
        
        Args:
            rng: RNG state or PRNG key
            shape: Shape of output array
            minval: Minimum value
            maxval: Maximum value
            
        Returns:
            Random array with uniform distribution
        """
        ...
    
    @abstractmethod
    def normal(self, rng: Any, shape: tuple, mean: float = 0.0, std: float = 1.0) -> Any:
        """
        Generate normal random samples.
        
        Args:
            rng: RNG state or PRNG key
            shape: Shape of output array
            mean: Mean value
            std: Standard deviation
            
        Returns:
            Random array with normal distribution
        """
        ...
    
    @abstractmethod
    def to_device(self, x: Any, device: str = None) -> Any:
        """
        Move tensor/array to specified device.
        
        Args:
            x: Tensor/array to move
            device: Target device ("cpu", "gpu", "webgpu")
            
        Returns:
            Tensor/array on target device
        """
        ...
    
    @abstractmethod
    def get_device(self) -> str:
        """
        Get current device.
        
        Returns:
            Device name ("cpu", "gpu", "webgpu")
        """
        ...
    
    def register_dynamics_kernel(self, key: str, fn: Callable) -> None:
        """
        Register a high-performance dynamics kernel (optional).
        
        This allows backends to provide optimized implementations of
        specific dynamics models (e.g., rigid body physics in Rust).
        
        Args:
            key: Unique identifier for the kernel
            fn: Kernel function
        """
        pass
    
    def register_constraint_kernel(self, key: str, fn: Callable) -> None:
        """
        Register a high-performance constraint kernel (optional).
        
        This allows backends to provide optimized implementations of
        specific constraint projections (e.g., contact constraints in Rust).
        
        Args:
            key: Unique identifier for the kernel
            fn: Kernel function
        """
        pass


# Import runtime backends (concrete implementations)
from enerdynamics.core.backends.runtime import (
    JaxBackend,
    NumpyBackend,
    TorchBackend,
    RuntimeBackendManager,
)

# Factory function for creating backends
def get_backend(name: str = "jax", device: str = "cpu", **kwargs) -> Backend:
    """
    Factory function to create a backend by name.
    
    Args:
        name: Backend name ("jax", "numpy", "torch", etc.)
        device: Device name ("cpu", "gpu", "webgpu")
        **kwargs: Additional backend-specific arguments
        
    Returns:
        Backend instance
        
    Raises:
        ValueError: If backend name is not recognized
        ImportError: If required backend library is not installed
    """
    backends = {
        "jax": JaxBackend,
        "numpy": NumpyBackend,
        "torch": TorchBackend,
    }
    
    if name not in backends:
        raise ValueError(
            f"Unknown backend: {name}. Available backends: {list(backends.keys())}"
        )
    
    backend_class = backends[name]
    return backend_class(device=device, **kwargs)
