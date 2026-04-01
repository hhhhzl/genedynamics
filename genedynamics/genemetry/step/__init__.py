"""
Constrained optimization steps for trajectory refinement.

Auto-registers available backends on import.
"""

from genedynamics.genemetry.step.agp import AgpStep

try:
    from genedynamics.genemetry.step.backends import agp_numpy  # noqa: F401
except ImportError:
    pass

__all__ = ["AgpStep"]
