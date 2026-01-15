"""
EB-MBD solver (Emerging-Barrier Model-Based Diffusion).

This package mirrors the structure of `solvers/single/edoc` (planner + backends),
but the core reverse-diffusion loop is adapted from the lightweight
`MBDSolver` for speed. The JAX backend lives in `backends/ebmbd_jax.py`.
"""

from .ebmbd import EBMBDSolver

__all__ = ["EBMBDSolver"]

