"""
Backend protocol/base for CFS-MBD solver.

CFS-MBD = MBD-style diffusion + Augmented Lagrangian objective + CFS-based per-step QP projection.
"""

from __future__ import annotations

from typing import Protocol, Any, Dict, List

from enerdynamics.core.types import Trajectory


class CFSMBDBackend(Protocol):
    def plan(self, x0: Any, rng_key: Any | None = None) -> Dict[str, Any]:
        ...

    def sample_trajectories(self, x0: Any, n_samples: int, rng_key: Any | None = None) -> List[Trajectory]:
        ...

