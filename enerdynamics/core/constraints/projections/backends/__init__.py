"""
Backend-specific implementations of CFS projection.
"""

# Import backend implementations to trigger registration
try:
    from . import cfs_jax  # noqa: F401
except ImportError:
    pass  # JAX backend may not be available

try:
    from . import cfs_numpy  # noqa: F401
except ImportError:
    pass  # NumPy backend may not be available

__all__ = []

