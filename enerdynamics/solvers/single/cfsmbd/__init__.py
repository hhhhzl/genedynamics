"""
CFS-MBD solver: MBD with Augmented Lagrangian objective and CFS-based per-step QP projection.
"""

from .cfsmbd import CFSMBDSolver

__all__ = ["CFSMBDSolver"]
