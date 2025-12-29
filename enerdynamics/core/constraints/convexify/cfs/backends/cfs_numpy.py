"""
NumPy backend for CFS convexifier.

This is the reference implementation using NumPy.
"""

from typing import Optional, Callable
import numpy as np

from enerdynamics.core.constraints.convexify.cfs.cfs import CFSConvexifier
from enerdynamics.core.constraints.core.types import (
    ScheduleState,
    ScheduleParams,
    ConvexConstraint,
)
from enerdynamics.core.constraints.core.registry import register
from enerdynamics.core.types import Trajectory


@register("convexifier", "cfs", "numpy")
class CFSNumpyConvexifier(CFSConvexifier):
    """
    NumPy backend for CFS convexifier.
    
    This is the reference implementation that other backends should match.
    """
    
    def build_constraints(
        self,
        ref: Trajectory,
        params: ScheduleParams,
        state: ScheduleState
    ) -> ConvexConstraint:
        """
        Build CFS constraints using NumPy.
        
        This implementation uses the base class method which already
        uses NumPy, so we just call the parent.
        """
        return super().build_constraints(ref, params, state)


