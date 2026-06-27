"""
Retraction operators for feasibility projection.

Auto-registers available backends on import.
"""

from genedynamics.genemetry.retraction.cfs import CfsRetraction
from genedynamics.genemetry.retraction.stepping import SteppingRetraction
from genedynamics.genemetry.retraction.local_cfs import LocalCfsRetraction

try:
    from genedynamics.genemetry.retraction.backends import cfs_jax  
    from genedynamics.genemetry.retraction.backends import stepping_jax  
except ImportError:
    pass

try:
    from genedynamics.genemetry.retraction.backends import local_cfs_numpy  
except ImportError:
    pass

__all__ = ["CfsRetraction", "SteppingRetraction", "LocalCfsRetraction"]
