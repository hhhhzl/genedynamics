"""
Backend protocol/base for MDOC solver.

MDOC = MBD diffusion driver + ConstraintFilter (action-space filtering).
"""

from __future__ import annotations

from typing import Protocol, Any, Dict, List

from genedynamics.core.types import Trajectory


class MDOCBackend(Protocol):
    def plan(self, x0: Any, rng_key: Any | None = None) -> Dict[str, Any]:
        ...

    def sample_trajectories(self, x0: Any, n_samples: int, rng_key: Any | None = None) -> List[Trajectory]:
        ...


