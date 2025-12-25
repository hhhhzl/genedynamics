"""
Constraint schedulers: Parameter annealing and adaptive tuning.

Schedulers generate ScheduleParams based on ScheduleState, controlling
how constraints are enforced throughout the optimization process.
"""

# Import to trigger registration
from . import cosine_anneal  # noqa: F401
from . import dual_anneal  # noqa: F401
from . import adaptive_gate  # noqa: F401
from .base import Scheduler
from .cosine_anneal import CosineAnnealScheduler
from .dual_anneal import DualAnnealScheduler
from .adaptive_gate import AdaptiveGateScheduler
from .presets import (
    create_preset_scheduler,
    soft_to_hard_scheduler,
    aggressive_scheduler,
    conservative_scheduler,
    adaptive_scheduler,
    dual_anneal_scheduler,
)

__all__ = [
    "Scheduler",
    "CosineAnnealScheduler",
    "DualAnnealScheduler",
    "AdaptiveGateScheduler",
    "create_preset_scheduler",
    "soft_to_hard_scheduler",
    "aggressive_scheduler",
    "conservative_scheduler",
    "adaptive_scheduler",
    "dual_anneal_scheduler",
]

