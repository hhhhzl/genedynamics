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
    
    name: str  # "jax", "torch", "rust", "wasm", "remote"
    
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


class JaxBackend:
    """
    JAX backend implementation.
    
    Provides JIT compilation, automatic differentiation, and efficient
    vectorization using JAX's vmap and jit decorators.
    """
    
    name = "jax"
    
    def __init__(self):
        try:
            import jax
            import jax.numpy as jnp
            self.jax = jax
            self.jnp = jnp
        except ImportError:
            raise ImportError(
                "JAX backend requires JAX to be installed. "
                "Install with: pip install jax jaxlib"
            )
    
    def tensor(self, x: Any, dtype: Optional[Any] = None) -> Any:
        """Convert to JAX array."""
        if dtype is None:
            return self.jnp.asarray(x)
        return self.jnp.asarray(x, dtype=dtype)
    
    def concat(self, xs: list, axis: int = 0) -> Any:
        """Concatenate JAX arrays."""
        return self.jnp.concatenate(xs, axis=axis)
    
    def randn(self, shape: tuple, rng_key: Optional[Any] = None) -> Any:
        """Generate random normal samples using JAX."""
        if rng_key is None:
            rng_key = self.jax.random.PRNGKey(0)
        return self.jax.random.normal(rng_key, shape)
    
    def vmap(self, fn: Callable) -> Callable:
        """Vectorize function using JAX vmap."""
        return self.jax.vmap(fn)
    
    def jit(self, fn: Callable) -> Callable:
        """JIT compile function using JAX."""
        return self.jax.jit(fn)


class NumpyBackend:
    """
    NumPy backend implementation (reference/fallback).
    
    This is a simple backend for testing and environments that don't
    require JIT compilation or automatic differentiation.
    """
    
    name = "numpy"
    
    def __init__(self):
        self.np = np
    
    def tensor(self, x: Any, dtype: Optional[Any] = None) -> Any:
        """Convert to NumPy array."""
        if dtype is None:
            return np.asarray(x)
        return np.asarray(x, dtype=dtype)
    
    def concat(self, xs: list, axis: int = 0) -> Any:
        """Concatenate NumPy arrays."""
        return np.concatenate(xs, axis=axis)
    
    def randn(self, shape: tuple, rng_key: Optional[Any] = None) -> Any:
        """Generate random normal samples using NumPy."""
        if rng_key is not None:
            np.random.seed(rng_key if isinstance(rng_key, int) else hash(rng_key) % 2**32)
        return np.random.randn(*shape)
    
    def vmap(self, fn: Callable) -> Callable:
        """Vectorize function (no-op for NumPy, just returns function)."""
        return fn
    
    def jit(self, fn: Callable) -> Callable:
        """JIT compile (no-op for NumPy, just returns function)."""
        return fn


class TorchBackend:
    """
    PyTorch backend implementation (stub for future implementation).
    
    This backend would provide PyTorch tensor operations and support
    for neural network-based dynamics and energy models.
    """
    
    name = "torch"
    
    def __init__(self):
        try:
            import torch
            self.torch = torch
        except ImportError:
            raise ImportError(
                "Torch backend requires PyTorch to be installed. "
                "Install with: pip install torch"
            )
    
    def tensor(self, x: Any, dtype: Optional[Any] = None) -> Any:
        """Convert to PyTorch tensor."""
        if dtype is None:
            return self.torch.as_tensor(x)
        return self.torch.as_tensor(x, dtype=dtype)
    
    def concat(self, xs: list, axis: int = 0) -> Any:
        """Concatenate PyTorch tensors."""
        return self.torch.cat(xs, dim=axis)
    
    def randn(self, shape: tuple, rng_key: Optional[Any] = None) -> Any:
        """Generate random normal samples using PyTorch."""
        if rng_key is not None:
            generator = self.torch.Generator()
            if isinstance(rng_key, int):
                generator.manual_seed(rng_key)
            else:
                generator.manual_seed(hash(rng_key) % 2**32)
            return self.torch.randn(shape, generator=generator)
        return self.torch.randn(shape)
    
    def vmap(self, fn: Callable) -> Callable:
        """Vectorize function (PyTorch doesn't have vmap, use manual batching)."""
        # PyTorch doesn't have vmap, so we return the function as-is
        # Users should manually batch operations
        return fn
    
    def jit(self, fn: Callable) -> Callable:
        """JIT compile function using TorchScript."""
        return self.torch.jit.script(fn)


# Factory function for creating backends
def get_backend(name: str = "jax") -> Backend:
    """
    Factory function to create a backend by name.
    
    Args:
        name: Backend name ("jax", "numpy", "torch", etc.)
        
    Returns:
        Backend instance
        
    Raises:
        ValueError: If backend name is not recognized
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
    
    return backends[name]()
