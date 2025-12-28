"""
Unified array interface for zero-copy backend switching.

This module provides BackendArray, a wrapper that:
1. Delays conversion until necessary (lazy evaluation)
2. Caches conversions to avoid repeated work
3. Supports zero-copy when possible
4. Provides unified interface across numpy/jax/torch/rust

Performance optimizations:
- Lazy conversion: Only convert when needed
- Conversion caching: Cache converted arrays
- Zero-copy when possible: Avoid copying if already in target backend
- Type preservation: Keep original backend when possible
"""

from typing import Union, Literal, Optional, Any
import numpy as np

# Optional imports with graceful fallback
try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jnp = None
    JAX_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# Type definitions
BackendType = Literal["numpy", "jax", "torch", "rust"]


class BackendArray:
    """
    Unified array interface for zero-copy backend switching.
    
    This class wraps arrays from different backends (numpy, JAX, PyTorch, Rust)
    and provides a unified interface with lazy conversion and caching.
    
    Design principles:
    1. Lazy conversion: Only convert when explicitly requested
    2. Conversion caching: Cache converted arrays to avoid repeated work
    3. Zero-copy when possible: Avoid copying if already in target backend
    4. Type preservation: Keep original backend when possible
    
    Performance benefits:
    - Eliminates redundant conversions in hot paths
    - Reduces memory allocations
    - Enables backend-agnostic code
    
    Example:
        >>> arr = BackendArray(np.array([1, 2, 3]), backend="numpy")
        >>> arr_jax = arr.to_jax()  # Converted and cached
        >>> arr_jax2 = arr.to_jax()  # Returns cached version (zero overhead)
    """
    
    def __init__(self, data: Any, backend: BackendType = "numpy"):
        """
        Initialize BackendArray.
        
        Args:
            data: Array data (numpy array, JAX array, PyTorch tensor, etc.)
            backend: Backend type ("numpy", "jax", "torch", "rust")
        """
        self._data = data
        self._backend = backend
        
        # Lazy conversion caches (populated on first conversion)
        self._numpy_data: Optional[np.ndarray] = None
        self._jax_data: Optional[Any] = None
        self._torch_data: Optional[Any] = None
    
    @property
    def backend(self) -> BackendType:
        """Get current backend type."""
        return self._backend
    
    @property
    def data(self) -> Any:
        """
        Get raw data in current backend.
        
        Returns:
            Raw array data (numpy array, JAX array, PyTorch tensor, etc.)
        """
        return self._data
    
    def to_numpy(self) -> np.ndarray:
        """
        Convert to NumPy array (lazy + cached).
        
        Returns:
            NumPy array
            
        Performance:
            - First call: Converts and caches
            - Subsequent calls: Returns cached version (zero overhead)
        """
        if self._numpy_data is not None:
            return self._numpy_data
        
        # Convert based on source backend
        if self._backend == "numpy":
            self._numpy_data = np.asarray(self._data, dtype=np.float32)
        elif self._backend == "jax" and JAX_AVAILABLE:
            # JAX to NumPy: zero-copy if possible (on CPU)
            self._numpy_data = np.asarray(self._data, dtype=np.float32)
        elif self._backend == "torch" and TORCH_AVAILABLE:
            # PyTorch to NumPy: requires detach and CPU transfer
            if self._data.requires_grad:
                self._data = self._data.detach()
            self._numpy_data = self._data.cpu().numpy().astype(np.float32)
        else:
            # Fallback: try to convert to numpy
            self._numpy_data = np.asarray(self._data, dtype=np.float32)
        
        return self._numpy_data
    
    def to_jax(self):
        """
        Convert to JAX array (lazy + cached).
        
        Returns:
            JAX array
            
        Raises:
            RuntimeError: If JAX is not available
            
        Performance:
            - First call: Converts and caches
            - Subsequent calls: Returns cached version (zero overhead)
        """
        if self._jax_data is not None:
            return self._jax_data
        
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not available. Install with: pip install jax jaxlib")
        
        # Convert based on source backend
        if self._backend == "jax":
            # Already JAX: zero-copy
            self._jax_data = self._data
        elif self._backend == "numpy":
            # NumPy to JAX: efficient conversion
            self._jax_data = jnp.asarray(self._data, dtype=jnp.float32)
        elif self._backend == "torch" and TORCH_AVAILABLE:
            # PyTorch to JAX: via NumPy (two-step conversion)
            numpy_data = self.to_numpy()
            self._jax_data = jnp.asarray(numpy_data, dtype=jnp.float32)
        else:
            # Fallback: via NumPy
            numpy_data = self.to_numpy()
            self._jax_data = jnp.asarray(numpy_data, dtype=jnp.float32)
        
        return self._jax_data
    
    def to_torch(self):
        """
        Convert to PyTorch tensor (lazy + cached).
        
        Returns:
            PyTorch tensor
            
        Raises:
            RuntimeError: If PyTorch is not available
            
        Performance:
            - First call: Converts and caches
            - Subsequent calls: Returns cached version (zero overhead)
        """
        if self._torch_data is not None:
            return self._torch_data
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. Install with: pip install torch")
        
        # Convert based on source backend
        if self._backend == "torch":
            # Already PyTorch: zero-copy
            self._torch_data = self._data
        elif self._backend == "numpy":
            # NumPy to PyTorch: efficient conversion (shares memory if possible)
            self._torch_data = torch.from_numpy(self._data).float()
        elif self._backend == "jax" and JAX_AVAILABLE:
            # JAX to PyTorch: via NumPy (two-step conversion)
            numpy_data = self.to_numpy()
            self._torch_data = torch.from_numpy(numpy_data).float()
        else:
            # Fallback: via NumPy
            numpy_data = self.to_numpy()
            self._torch_data = torch.from_numpy(numpy_data).float()
        
        return self._torch_data
    
    def shape(self) -> tuple:
        """Get array shape (backend-agnostic)."""
        if self._backend == "numpy":
            return self._data.shape
        elif self._backend == "jax" and JAX_AVAILABLE:
            return self._data.shape
        elif self._backend == "torch" and TORCH_AVAILABLE:
            return tuple(self._data.shape)
        else:
            # Fallback: try to get shape
            return getattr(self._data, 'shape', ())
    
    def dtype(self):
        """Get array dtype (backend-agnostic)."""
        if self._backend == "numpy":
            return self._data.dtype
        elif self._backend == "jax" and JAX_AVAILABLE:
            return self._data.dtype
        elif self._backend == "torch" and TORCH_AVAILABLE:
            return self._data.dtype
        else:
            # Fallback
            return getattr(self._data, 'dtype', None)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"BackendArray(backend={self._backend}, shape={self.shape()}, dtype={self.dtype()})"
    
    @classmethod
    def ensure_backend(cls, data: Any, target_backend: BackendType) -> "BackendArray":
        """
        Ensure data is in target backend (zero-copy if possible).
        
        This is a convenience method that creates a BackendArray and converts
        to target backend if needed.
        
        Args:
            data: Array data (any backend or BackendArray)
            target_backend: Target backend type
            
        Returns:
            BackendArray in target backend
            
        Performance:
            - If data is already BackendArray in target backend: zero-copy
            - Otherwise: creates new BackendArray and converts (lazy)
        """
        # If already BackendArray in target backend, return as-is
        if isinstance(data, BackendArray):
            if data.backend == target_backend:
                return data
            # Convert to target backend
            if target_backend == "numpy":
                return cls(data.to_numpy(), "numpy")
            elif target_backend == "jax":
                return cls(data.to_jax(), "jax")
            elif target_backend == "torch":
                return cls(data.to_torch(), "torch")
            else:
                # Fallback: via numpy
                return cls(data.to_numpy(), target_backend)
        
        # Create new BackendArray (detect backend from data)
        detected_backend = cls._detect_backend(data)
        arr = cls(data, detected_backend)
        
        # Convert to target if needed
        if detected_backend != target_backend:
            if target_backend == "numpy":
                return cls(arr.to_numpy(), "numpy")
            elif target_backend == "jax":
                return cls(arr.to_jax(), "jax")
            elif target_backend == "torch":
                return cls(arr.to_torch(), "torch")
        
        return arr
    
    @staticmethod
    def _detect_backend(data: Any) -> BackendType:
        """
        Detect backend from data type.
        
        Args:
            data: Array data
            
        Returns:
            Detected backend type
        """
        # Check for JAX
        if JAX_AVAILABLE:
            try:
                import jax
                if isinstance(data, jax.Array) or (hasattr(data, '__class__') and 'jax' in str(type(data))):
                    return "jax"
            except:
                pass
        
        # Check for PyTorch
        if TORCH_AVAILABLE:
            if isinstance(data, torch.Tensor):
                return "torch"
        
        # Check for NumPy
        if isinstance(data, np.ndarray):
            return "numpy"
        
        # Default to numpy (will convert on first use)
        return "numpy"


# Convenience function
def ensure_backend(data: Any, target_backend: BackendType) -> BackendArray:
    """
    Convenience function to ensure data is in target backend.
    
    Args:
        data: Array data (any backend or BackendArray)
        target_backend: Target backend type
        
    Returns:
        BackendArray in target backend
    """
    return BackendArray.ensure_backend(data, target_backend)


