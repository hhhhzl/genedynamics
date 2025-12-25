"""
NumPy backend implementation (reference/fallback).

This is a simple backend for testing and environments that don't
require JIT compilation or automatic differentiation.
Only supports CPU device.
"""

from typing import Any, Callable, Optional
import numpy as np


class NumpyBackend:
    """
    NumPy backend implementation (reference/fallback).
    
    This is a simple backend for testing and environments that don't
    require JIT compilation or automatic differentiation.
    Only supports CPU device.
    """
    
    name = "numpy"
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize NumPy backend.
        
        Args:
            device: Device to use (only "cpu" is supported)
            
        Raises:
            ValueError: If device is not "cpu"
        """
        if device != "cpu":
            raise ValueError(
                f"NumPy backend only supports CPU device, got {device}"
            )
        self.device = "cpu"
        self.np = np
        self._rng_state = None
    
    def tensor(self, x: Any, dtype: Optional[Any] = None) -> Any:
        """
        Convert input to NumPy array.
        
        Args:
            x: Input data (list, array, etc.)
            dtype: Optional data type
            
        Returns:
            NumPy array
        """
        if dtype is None:
            return np.asarray(x)
        return np.asarray(x, dtype=dtype)
    
    def concat(self, xs: list, axis: int = 0) -> Any:
        """
        Concatenate NumPy arrays.
        
        Args:
            xs: List of arrays to concatenate
            axis: Axis along which to concatenate
            
        Returns:
            Concatenated array
        """
        return np.concatenate(xs, axis=axis)
    
    def randn(self, shape: tuple, rng_key: Optional[Any] = None) -> Any:
        """
        Generate random normal samples using NumPy.
        
        Args:
            shape: Shape of output array
            rng_key: Optional random seed (int)
            
        Returns:
            Random array
        """
        if rng_key is not None:
            np.random.seed(rng_key if isinstance(rng_key, int) else hash(rng_key) % 2**32)
        return np.random.randn(*shape)
    
    def vmap(self, fn: Callable) -> Callable:
        """
        Vectorize function (no-op for NumPy, just returns function).
        
        Args:
            fn: Function to vectorize
            
        Returns:
            Same function (NumPy doesn't have vmap)
        """
        return fn
    
    def jit(self, fn: Callable) -> Callable:
        """
        JIT compile (no-op for NumPy, just returns function).
        
        Args:
            fn: Function to compile
            
        Returns:
            Same function (NumPy doesn't have JIT)
        """
        return fn
    
    def create_rng(self, seed: int) -> int:
        """
        Create a random number generator state from seed.
        
        For NumPy, this just returns the seed value.
        
        Args:
            seed: Random seed
            
        Returns:
            Seed value (int)
        """
        np.random.seed(seed)
        return seed
    
    def split_rng(self, rng: Any, num: int = 2) -> Any:
        """
        Split RNG state (no-op for NumPy, returns same seed).
        
        Args:
            rng: RNG state (seed)
            num: Number of states to create (ignored)
            
        Returns:
            Same seed (NumPy doesn't need splitting)
        """
        return rng
    
    def uniform(self, rng: Any, shape: tuple, minval: float = 0.0, maxval: float = 1.0) -> Any:
        """
        Generate uniform random samples.
        
        Args:
            rng: Random seed (int)
            shape: Shape of output array
            minval: Minimum value
            maxval: Maximum value
            
        Returns:
            Random array with uniform distribution
        """
        seed = rng if isinstance(rng, int) else hash(rng) % 2**32
        np.random.seed(seed)
        return np.random.uniform(minval, maxval, size=shape)
    
    def normal(self, rng: Any, shape: tuple, mean: float = 0.0, std: float = 1.0) -> Any:
        """
        Generate normal random samples.
        
        Args:
            rng: Random seed (int)
            shape: Shape of output array
            mean: Mean value
            std: Standard deviation
            
        Returns:
            Random array with normal distribution
        """
        seed = rng if isinstance(rng, int) else hash(rng) % 2**32
        np.random.seed(seed)
        return np.random.normal(mean, std, size=shape)
    
    def to_device(self, x: Any, device: str = None) -> Any:
        """
        Move array to specified device (no-op for NumPy).
        
        Args:
            x: Array to move
            device: Target device (ignored, NumPy only supports CPU)
            
        Returns:
            Array (unchanged)
        """
        return x
    
    def get_device(self) -> str:
        """
        Get current device.
        
        Returns:
            Always "cpu" for NumPy
        """
        return "cpu"
    
    def register_dynamics_kernel(self, key: str, fn: Callable) -> None:
        """
        Register a high-performance dynamics kernel (optional).
        
        Args:
            key: Unique identifier for the kernel
            fn: Kernel function
        """
        # NumPy doesn't need explicit kernel registration
        pass
    
    def register_constraint_kernel(self, key: str, fn: Callable) -> None:
        """
        Register a high-performance constraint kernel (optional).
        
        Args:
            key: Unique identifier for the kernel
            fn: Kernel function
        """
        # NumPy doesn't need explicit kernel registration
        pass

