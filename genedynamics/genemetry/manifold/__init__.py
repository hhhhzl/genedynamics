"""
Constraint manifold implementations.

Auto-registers available backends on import.
"""

from genedynamics.genemetry.manifold.sdf import SdfManifold

try:
    from genedynamics.genemetry.manifold.backends import sdf_jax  
except ImportError:
    pass

try:
    from genedynamics.genemetry.manifold.backends import sdf_torch
except ImportError:
    pass

__all__ = ["SdfManifold"]
