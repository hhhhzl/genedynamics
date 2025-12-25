"""
EDOC solver implementation.

Energy-Driven Optimal Control using diffusion + A-MCSA + ADM.
"""

# Import backend implementations to trigger registration
try:
    from . import backends  # noqa: F401
except ImportError:
    pass  # Backend implementations may not be available

from .edoc import (
    EDOCPlanner,
    EDOCSolver,
    EnergyToLegacyAdapter,
    run_edoc,
)

__all__ = [
    'EDOCPlanner',
    'EDOCSolver',
    'EnergyToLegacyAdapter',
    'run_edoc',
]

