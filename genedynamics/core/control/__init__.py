"""Upstream control-action primitives (solver-agnostic).

Currently the position-stiffness primitive `u=(r,K,nu)` with the log-Euclidean
SPD chart `K=exp(S)` — reusable by any env (impedance law) or solver (MDAC and
beyond). See `stiffness.py`.
"""

from genedynamics.core.control.stiffness import (
    PrimitiveSpec,
    metric_GK_diag,
    metric_GK_diag_full,
    stiffness_log_to_pd,
    stiffness_pd_to_log,
    svec2sym,
    svec_len,
    sym2svec,
    unpack_primitive,
)

__all__ = [
    "PrimitiveSpec",
    "metric_GK_diag",
    "metric_GK_diag_full",
    "stiffness_log_to_pd",
    "stiffness_pd_to_log",
    "svec2sym",
    "svec_len",
    "sym2svec",
    "unpack_primitive",
]
