"""
Projection operators for hard constraint enforcement.
"""

# Import backend implementations to trigger registration
try:
    from enerdynamics.core.constraints.legacy.projections import backends  # noqa: F401
except ImportError:
    pass  # Backends may not be available

from enerdynamics.core.constraints.legacy.projections.cfs import CFSProjection

__all__ = ["CFSProjection"]
