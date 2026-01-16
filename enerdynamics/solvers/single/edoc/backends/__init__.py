"""
Backend implementations for EDOC solver.

This package contains backend-specific implementations of the EDOC reverse diffusion algorithm.
"""

# Import backend implementations to trigger registration
try:
    from . import edoc_jax  
    from . import edoc_numpy  
except ImportError:
    pass  # Backend implementations may not be available


