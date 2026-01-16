"""
Constraint schedulers: Parameter annealing and adaptive tuning.

Schedulers generate ScheduleParams based on ScheduleState, controlling
how constraints are enforced throughout the optimization process.
"""

# Import to trigger registration
from . import cosine_anneal  
from . import dual_anneal  
from . import adaptive_gate  

# Base classes
from .base import Scheduler

# Legacy schedulers
from .cosine_anneal import CosineAnnealScheduler
from .dual_anneal import DualAnnealScheduler
from .adaptive_gate import AdaptiveGateScheduler

# New constraint schedulers (import to trigger registration)
from . import ConstraintScheduler  
from .ConstraintScheduler import (
    ConstraintScheduler as ConstraintSchedulerBase,
    FixedConstraintScheduler,
    DualControlConstraintScheduler,
)

# New diffusion schedulers (import to trigger registration)
from . import DiffusionScheduler  
from .DiffusionScheduler import (
    DiffusionScheduler as DiffusionSchedulerBase,
    FixedDiffusionScheduler,
    DualControlDiffusionScheduler,
)

# Composite scheduler (import to trigger registration)
from . import CompositeScheduler  
from .CompositeScheduler import CompositeScheduler, MergeStrategy

# Utilities
from .utils import DiffusionNoiseSchedule

# Presets
from .presets import (
    create_preset_scheduler,
    soft_to_hard_scheduler,
    aggressive_scheduler,
    conservative_scheduler,
    adaptive_scheduler,
    dual_anneal_scheduler,
)

__all__ = [
    # Base
    "Scheduler",
    
    # Legacy schedulers
    "CosineAnnealScheduler",
    "DualAnnealScheduler",
    "AdaptiveGateScheduler",
    
    # New constraint schedulers
    "ConstraintSchedulerBase",
    "FixedConstraintScheduler",
    "DualControlConstraintScheduler",
    
    # New diffusion schedulers
    "DiffusionSchedulerBase",
    "FixedDiffusionScheduler",
    "DualControlDiffusionScheduler",
    
    # Composite scheduler
    "CompositeScheduler",
    "MergeStrategy",
    
    # Utilities
    "DiffusionNoiseSchedule",
    
    # Presets
    "create_preset_scheduler",
    "soft_to_hard_scheduler",
    "aggressive_scheduler",
    "conservative_scheduler",
    "adaptive_scheduler",
    "dual_anneal_scheduler",
]

