"""
Backend implementations for EB-MBD solver.
JAX backend is preferred; a NumPy backend is also provided for debugging.
"""

# Import to trigger registry registration
try:
    from . import ebmbd_jax  
except ImportError:
    pass

try:
    from . import ebmbd_numpy  
except ImportError:
    pass

