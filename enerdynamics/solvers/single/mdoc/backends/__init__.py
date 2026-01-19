"""
MDOC backend implementations.
"""

from .mdoc_jax import MDOCBackendJax
from .mdoc_numpy import MDOCBackendNumpy

__all__ = ["MDOCBackendJax", "MDOCBackendNumpy"]


