"""
Backend implementations for EB-MBD solver.
JAX backend is preferred; a NumPy backend is also provided for debugging.
"""

# Import to trigger registry registration
try:
    from . import ebmbd_jax  # noqa: F401
except ImportError:
    pass

try:
    from . import ebmbd_numpy  # noqa: F401
except ImportError:
    pass

