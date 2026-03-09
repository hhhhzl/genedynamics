"""
Backend protocol adapter for MBD3D solver.
"""

from __future__ import annotations

from typing import Any

from genedynamics.solvers.common.model_based_diffusion.backend_protocol import ensure_unified_backend


def to_unified_backend(backend_impl: Any) -> Any:
    """Expose unified plan/plan_batch interface for BaseModelBasedDiffusionSolver."""
    return ensure_unified_backend(backend_impl)
