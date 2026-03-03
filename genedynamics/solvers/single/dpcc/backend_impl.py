"""
Backend protocol/base for DPCC solver.
"""

from __future__ import annotations

from typing import Protocol, Any, Dict, List

from genedynamics.core.types import Trajectory


class DPCCBackend(Protocol):
    def plan(self, x0: Any | None = None, rng_key: Any | None = None) -> Dict[str, Any]:
        ...

    def sample_trajectories(
        self, x0: Any | None, n_samples: int, rng_key: Any | None = None
    ) -> List[Trajectory]:
        ...
