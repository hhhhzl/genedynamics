"""
NumPy backend for 2GO constraint scheduler.
"""

from typing import Dict, Any

from genedynamics.core.constraints.schedulers.ConstraintScheduler.twogo.twogo import TwoGOConstraintScheduler
from genedynamics.core.constraints.core.types import ScheduleState
from genedynamics.core.constraints.core.registry import register


@register("scheduler", "twogo_constraint", "numpy")
class TwoGOConstraintSchedulerNumpy(TwoGOConstraintScheduler):
    """
    NumPy backend for 2GO constraint scheduler.
    """

    def constraint_params(self, state: ScheduleState, record: bool = True) -> Dict[str, Any]:
        return super().constraint_params(state, record=record)

