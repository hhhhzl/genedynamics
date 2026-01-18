"""
Action-space constraint filters (new architecture).

These are lightweight, backend-friendly filters that take in an action sequence
and return a modified action sequence that better satisfies constraints.

MDOC uses this abstraction so we can ablate:
- closed-form per-step CBF projection
- QP-based per-step CBF filtering
"""

from .base import ConstraintFilter
from .noop import NoOpConstraintFilter
from .cbf_closed_form import ClosedFormCBFFilter
from .cbf_qp import QPBasedCBFFilter

__all__ = ["ConstraintFilter", "NoOpConstraintFilter", "ClosedFormCBFFilter", "QPBasedCBFFilter"]
