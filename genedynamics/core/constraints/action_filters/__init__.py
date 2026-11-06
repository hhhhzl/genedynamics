"""
Action-space constraint filters (new architecture).

These are lightweight, backend-friendly filters that take in an action sequence
and return a modified action sequence that better satisfies constraints.

MDOC uses this abstraction so we can ablate:
- closed-form per-step CBF projection
- QP-based per-step CBF filtering
"""

from importlib import import_module

from .base import ConstraintFilter
from .noop import NoOpConstraintFilter
from .trajectory_feasibility_torch import TorchTrajectoryFeasibilityFilter

_OPTIONAL_JAX_FILTERS = {
    "CFSQPFullFilter": ".cfs_qp_full",
    "CFSQPPerStepFilter": ".cfs_qp_perstep",
    "ClosedFormCBFFilter": ".cbf_closed_form",
    "ClosedFormCBFFilterJointLift": ".cbf_closed_form_joint_lift",
    "CorridorCBFFilter": ".task_specific.cbf_corridor",
    "QPBasedCBFFilter": ".cbf_qp",
    "QPBasedCBFFilterJointLift": ".cbf_qp_joint_lift",
}


def __getattr__(name):
    """Load JAX filters only when a caller explicitly requests one."""

    module_name = _OPTIONAL_JAX_FILTERS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value

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
