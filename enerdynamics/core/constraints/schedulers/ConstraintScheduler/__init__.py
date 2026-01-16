"""
Constraint schedulers.

Constraint schedulers are responsible for constraint-related parameters:
- rho (slack penalty)
- topK (active constraint count)
- eps (solver tolerance)
- I_QP (QP iterations)
- qp_gate, qp_prob (QP invocation)
"""

# Import to trigger registration
from . import base
from . import fixed
from . import dualcontrol
from . import emergingbarrier

from .base import ConstraintScheduler
from .fixed import FixedConstraintScheduler
from .dualcontrol import DualControlConstraintScheduler
from .emergingbarrier import EmergingBarrierConstraintScheduler

__all__ = [
    "ConstraintScheduler",
    "FixedConstraintScheduler",
    "DualControlConstraintScheduler",
    "EmergingBarrierConstraintScheduler",
]

