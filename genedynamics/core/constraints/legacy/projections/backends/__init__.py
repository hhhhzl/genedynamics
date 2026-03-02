"""
Backend-specific implementations of CFS projection.
"""

# Import backend implementations to trigger registration
try:
    from . import cfs_jax  
except ImportError:
    pass  # JAX backend may not be available

try:
    from . import cfs_numpy  
except ImportError:
    pass  # NumPy backend may not be available

__all__ = []

