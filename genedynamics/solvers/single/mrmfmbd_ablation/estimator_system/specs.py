"""EstimatorConfig — multi-fidelity control-variate score estimator config.

Frozen dataclass + factory, mirroring fidelity_system/specs.py and
mode_system/specs.py. This is the central ML sub-contribution: instead of
picking ONE simulator fidelity per diffusion step (the legacy ladder), the
MBD score is estimated by combining a cheap low-fidelity reward over ALL M
proposals with an expensive high-fidelity reward over a SUBSET, using the
low-fidelity reward as a control variate. At a fixed per-step compute budget
this keeps the proposal pool large (low MC variance) while debiasing toward
the high-fidelity objective.

The estimator is task-agnostic: it consumes reward arrays + costs only, so the
same module is reusable on the trajectory-opt generality testbed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class EstimatorConfig:
    """Config for the per-diffusion-step multi-fidelity score estimator.

    Fields
    ------
    method        : "control_variate" (low-fi all + hi-fi subset) or
                    "single_fidelity" (legacy: one fidelity for all proposals).
    c_lo, c_hi    : relative cost of a single low-/high-fidelity evaluation
                    (match fidelity_system FidelityLevelSpec.cost).
    budget_per_step : B̄, the target compute budget per diffusion step.
    subset_frac   : fallback K/M when not budget-driven.
    eta_dual      : dual-variable ν step size (report Eq 23).
    nu0           : initial ν.
    """

    method: str = "control_variate"
    c_lo: float = 1.0
    c_hi: float = 7.0
    budget_per_step: float = 200.0
    subset_frac: float = 0.25
    eta_dual: float = 0.1
    nu0: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


def default_estimator_config() -> EstimatorConfig:
    """Defaults aligned with the 3-level fidelity ladder (cost 1 / 7 endpoints)."""
    return EstimatorConfig()
