"""
Dynamics linearization utilities.

This module provides utilities for linearizing dynamics to compute
Jacobians for state-to-control sensitivity analysis.
"""

from .autodiff import LinearizeDynamics

__all__ = ["LinearizeDynamics"]


