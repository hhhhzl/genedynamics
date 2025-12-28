"""
Backend implementations for EDOC solver.

This package contains backend-specific implementations of the EDOC reverse diffusion algorithm.
"""

# Import backend implementations to trigger registration
try:
    from . import edoc_jax  # noqa: F401
    from . import edoc_numpy  # noqa: F401
except ImportError:
    pass  # Backend implementations may not be available


