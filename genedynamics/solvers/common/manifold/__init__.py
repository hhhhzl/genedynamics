"""
Shared manifold utilities for geometry-aware solvers.
"""

from .jax_ops import build_active_rows, project_complement_batch

__all__ = [
    "build_active_rows",
    "project_complement_batch",
]

