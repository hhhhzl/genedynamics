"""
NumPy backend for EmergingBarrierConstraintScheduler.
"""

from typing import Dict, Any
from genedynamics.core.constraints.core.types import ScheduleState
from genedynamics.core.constraints.core.registry import register
from ..emergingbarrier import EmergingBarrierConstraintScheduler


@register("scheduler", "emerging_barrier", "numpy")
class EmergingBarrierConstraintSchedulerNumpy(EmergingBarrierConstraintScheduler):
    """NumPy backend (same as base, kept for parity and registry)."""

    def constraint_params(self, state: ScheduleState) -> Dict[str, Any]:
        return self._cached_params.copy()

