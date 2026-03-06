"""
NumPy backend for 2GO diffusion scheduler.
"""

from typing import Dict, Any

from genedynamics.core.constraints.schedulers.DiffusionScheduler.twogo.twogo import TwoGODiffusionScheduler
from genedynamics.core.constraints.core.types import ScheduleState
from genedynamics.core.constraints.core.registry import register


@register("scheduler", "twogo_diffusion", "numpy")
class TwoGODiffusionSchedulerNumpy(TwoGODiffusionScheduler):
    """
    NumPy backend for 2GO diffusion scheduler.
    """

    def diffusion_params(self, state: ScheduleState) -> Dict[str, Any]:
        return super().diffusion_params(state)

