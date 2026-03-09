"""
MRMFMBD (Soft-robot S1+S3) solver.

Multi-Resolution, Multi-Fidelity Model-Based Diffusion.
"""

from .mrmfmbd import MRMFMBDSolver
from .types import FidelityLevel, ModeRegime, MRMFMBDResult
from .protocols import FidelitySimulator, ModeMarginalizer

__all__ = [
    "MRMFMBDSolver",
    "FidelityLevel",
    "ModeRegime",
    "MRMFMBDResult",
    "FidelitySimulator",
    "ModeMarginalizer",
]
