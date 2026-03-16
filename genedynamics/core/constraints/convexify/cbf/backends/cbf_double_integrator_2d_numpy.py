"""
NumPy backend for CBF convexifier (double integrator 2D).

This is the reference implementation using NumPy.
"""

from typing import Optional
import numpy as np

from genedynamics.core.constraints.convexify.cbf.cbf import CBFConvexifier
from genedynamics.core.constraints.core.types import (
    ScheduleState,
    ScheduleParams,
    ConvexConstraint,
)
from genedynamics.core.constraints.core.registry import register


@register("convexifier", "cbf", "numpy")
class CBFNumpyConvexifier(CBFConvexifier):
    """
    NumPy backend for CBF convexifier (double integrator 2D).
    
    This is the reference implementation that other backends should match.
    """
    
    def build_constraints(
        self,
        ref,
        params: ScheduleParams,
        state: ScheduleState
    ) -> ConvexConstraint:
        """
        Build CBF constraints using NumPy.
        
        This implementation uses the base class method which already
        uses NumPy, so we just call the parent.
        """
        return super().build_constraints(ref, params, state)
