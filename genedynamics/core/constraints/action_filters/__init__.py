"""
Action-space constraint filters (new architecture).

These are lightweight, backend-friendly filters that take in an action sequence
and return a modified action sequence that better satisfies constraints.

MDOC uses this abstraction so we can ablate:
- closed-form per-step CBF projection
- QP-based per-step CBF filtering
"""

from .base import ConstraintFilter
from .cbf_closed_form import ClosedFormCBFFilter
from .cbf_closed_form_joint_lift import ClosedFormCBFFilterJointLift
from .cbf_qp import QPBasedCBFFilter
from .cbf_qp_joint_lift import QPBasedCBFFilterJointLift
from .cfs_qp_full import CFSQPFullFilter
from .cfs_qp_perstep import CFSQPPerStepFilter
from .noop import NoOpConstraintFilter
from .task_specific.cbf_corridor import CorridorCBFFilter
from .trajectory_feasibility_torch import TorchTrajectoryFeasibilityFilter

__all__ = [
    "CFSQPFullFilter",
    "CFSQPPerStepFilter",
    "ClosedFormCBFFilter",
    "ClosedFormCBFFilterJointLift",
    "ConstraintFilter",
    "CorridorCBFFilter",
    "NoOpConstraintFilter",
    "QPBasedCBFFilter",
    "QPBasedCBFFilterJointLift",
    "TorchTrajectoryFeasibilityFilter",
]
