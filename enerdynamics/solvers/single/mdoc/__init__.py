"""
MDOC solver package.

MDOC is modeled as:
  MBD-style diffusion driver + ConstraintFilter (action-space filtering per diffusion step)
"""

from .mdoc import MDOCSolver, run_mdoc

__all__ = ["MDOCSolver", "run_mdoc"]


