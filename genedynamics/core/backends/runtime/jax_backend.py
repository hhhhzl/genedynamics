"""
JAX backend implementation with device support.

This backend provides JIT compilation, automatic differentiation, and efficient
vectorization using JAX's vmap and jit decorators. Supports CPU and GPU devices.
"""

from typing import Any, Callable, Optional, Tuple
import os
import numpy as np


def _configure_jax_memory_defaults() -> None:
    # Respect explicit env overrides while keeping on-demand allocation as default.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")


_configure_jax_memory_defaults()

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None


class JaxBackend:
    """
    JAX backend implementation.
    
    Provides JIT compilation, automatic differentiation, and efficient
    vectorization using JAX's vmap and jit decorators.
    Supports CPU and GPU devices.
    """
    
    name = "jax"
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize JAX backend.
        
        Args:
            device: Device to use ("cpu" or "gpu")
            
        Raises:
            ImportError: If JAX is not installed
            RuntimeError: If GPU is requested but not available
        """
        if not HAS_JAX:
            raise ImportError(
                "JAX backend requires JAX to be installed. "
                "Install with: pip install jax jaxlib"
            )
        
        self.jax = jax
        self.jnp = jnp
        self.device = device
        
        # Set up device
        if device == "gpu" or device == "cuda":
            try:
                devices = jax.devices("gpu")
                if len(devices) == 0:
                    print("Warning: No GPU devices found, falling back to CPU")
                    self.device = "cpu"
                    self.devices = jax.devices("cpu")
                else:
                    print(f"Using GPU devices: {devices}")
                    self.devices = devices
            except Exception as e:
                print(f"Warning: GPU setup failed ({e}), falling back to CPU")
                self.device = "cpu"
                self.devices = jax.devices("cpu")
        else:
            print(f"Using CPU devices")
            self.devices = jax.devices("cpu")
            self.device = "cpu"
    
    def tensor(self, x: Any, dtype: Optional[Any] = None) -> Any:
        """
        Convert input to JAX array on the configured device.
        
        Args:
            x: Input data (numpy array, list, etc.)
            dtype: Optional data type
            
        Returns:
            JAX array on the configured device
        """
        if dtype is None:
            arr = self.jnp.asarray(x)
        else:
            arr = self.jnp.asarray(x, dtype=dtype)
        
        # Place on the configured device
        return self.to_device(arr)
    
    def concat(self, xs: list, axis: int = 0) -> Any:
        """
        Concatenate JAX arrays.
        
        Args:
            xs: List of arrays to concatenate
            axis: Axis along which to concatenate
            
        Returns:
            Concatenated array
        """
        return self.jnp.concatenate(xs, axis=axis)
    
    def randn(self, shape: tuple, rng_key: Optional[Any] = None) -> Any:
        """
        Generate random normal samples using JAX.
        
        Args:
            shape: Shape of output array
            rng_key: Optional PRNG key
            
        Returns:
            Random array on the configured device
        """
        if rng_key is None:
            rng_key = self.jax.random.PRNGKey(0)
        arr = self.jax.random.normal(rng_key, shape)
        return self.to_device(arr)
    
    def vmap(self, fn: Callable) -> Callable:
        """
        Vectorize function using JAX vmap.
        
        Args:
            fn: Function to vectorize
            
        Returns:
            Vectorized function
        """
        return self.jax.vmap(fn)
    
    def jit(self, fn: Callable) -> Callable:
        """
        JIT compile function using JAX.
        
        Args:
            fn: Function to compile
            
        Returns:
            Compiled function
        """
        return self.jax.jit(fn)
    
    def create_rng(self, seed: int) -> Any:
        """
        Create a PRNG key from seed.
        
        Args:
            seed: Random seed
            
        Returns:
            JAX PRNG key
        """
        return self.jax.random.PRNGKey(seed)
    
    def split_rng(self, rng: Any, num: int = 2) -> Any:
        """
        Split PRNG key into multiple keys.
        
        Args:
            rng: PRNG key to split
            num: Number of keys to create
            
        Returns:
            Tuple of PRNG keys
        """
        return self.jax.random.split(rng, num)
    
    def uniform(self, rng: Any, shape: tuple, minval: float = 0.0, maxval: float = 1.0) -> Any:
        """
        Generate uniform random samples.
        
        Args:
            rng: PRNG key
            shape: Shape of output array
            minval: Minimum value
            maxval: Maximum value
            
        Returns:
            Random array with uniform distribution on the configured device
        """
        arr = self.jax.random.uniform(rng, shape, minval=minval, maxval=maxval)
        return self.to_device(arr)
    
    def normal(self, rng: Any, shape: tuple, mean: float = 0.0, std: float = 1.0) -> Any:
        """
        Generate normal random samples.
        
        Args:
            rng: PRNG key
            shape: Shape of output array
            mean: Mean value
            std: Standard deviation
            
        Returns:
            Random array with normal distribution on the configured device
        """
        arr = self.jax.random.normal(rng, shape) * std + mean
        return self.to_device(arr)
    
    def to_device(self, x: Any, device: str = None) -> Any:
        """
        Move array to specified device.
        
        For JAX, arrays are placed on devices using jax.device_put.
        If device is None, uses the backend's configured device.
        
        Args:
            x: Array to move
            device: Target device ("cpu" or "gpu"). If None, uses backend device.
            
        Returns:
            Array on target device
        """
        if device is None:
            device = self.device
        
        # Get target device object
        if device == "gpu":
            try:
                target_devices = self.jax.devices("gpu")
                if len(target_devices) > 0:
                    target_device = target_devices[0]
                else:
                    target_device = self.jax.devices("cpu")[0]
            except Exception:
                target_device = self.jax.devices("cpu")[0]
        else:
            target_device = self.jax.devices("cpu")[0]
        
        return self.jax.device_put(x, target_device)
    
    def get_device(self) -> str:
        """
        Get current device.
        
        Returns:
            Device name ("cpu" or "gpu")
        """
        return self.device
    
    def register_dynamics_kernel(self, key: str, fn: Callable) -> None:
        """
        Register a high-performance dynamics kernel (optional).
        
        Args:
            key: Unique identifier for the kernel
            fn: Kernel function
        """
        # JAX doesn't need explicit kernel registration
        pass
    
    def register_constraint_kernel(self, key: str, fn: Callable) -> None:
        """
        Register a high-performance constraint kernel (optional).
        
        Args:
            key: Unique identifier for the kernel
            fn: Kernel function
        """
        # JAX doesn't need explicit kernel registration
        pass

