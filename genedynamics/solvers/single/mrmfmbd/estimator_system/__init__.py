"""Multi-fidelity control-variate score estimator for MRMFMBD (writeup §, central
ML sub-contribution).

Task-agnostic (reward arrays + costs only), so it is reusable on any
multi-fidelity reward oracle (soft-robot co-design AND the trajectory-opt
generality testbed). Mirrors the fidelity_system / mode_system subpackage
layout (specs.py + logic modules + __init__ exports).
"""

from __future__ import annotations

from .specs import EstimatorConfig, default_estimator_config
from .control_variate import (
    corrected_rewards,
    softmax_weights,
    weighted_mean,
    cv_score_weighted_mean,
    estimator_diagnostics,
)
from .budget import (
    realized_cost,
    optimal_subset_size,
    single_fidelity_pool,
    BudgetDual,
)

__all__ = [
    "EstimatorConfig",
    "default_estimator_config",
    "corrected_rewards",
    "softmax_weights",
    "weighted_mean",
    "cv_score_weighted_mean",
    "estimator_diagnostics",
    "realized_cost",
    "optimal_subset_size",
    "single_fidelity_pool",
    "BudgetDual",
]
