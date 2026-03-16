"""
CFS convexifier backends.

Backend implementations for CFS convexifier:
- cfs_numpy: NumPy backend
- cfs_jax: JAX backend (with JIT and vmap)
"""

# Import backend implementations to trigger registration
from . import cfs_numpy  
from . import cfs_jax  

__all__ = []


