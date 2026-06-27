"""MDAC solver — Manifold Diffusion Annealing Control (multi-backend).

DIAL's MBD annealing made manifold-aware: a receding-horizon sampling controller
over a lower-control primitive `U=(r,K,nu)`, with soft-feasibility weighting,
constraint-manifold tangent denoising + retraction, an optional model-free RL
prior, and a coupled DDPM/DDIM/flow annealing schedule
(`docs/mdac/MDAC_BUILD_PLAN.md`, `idea.txt`).

Composes the config surface, registry entry, the `core/` component math, and
the reverse step over the receding-horizon bridge.
"""

from genedynamics.solvers.single.mdac.mdac import MDACSolver
from genedynamics.solvers.single.mdac.backend_impl import (
    MdacBackend,
    to_unified_backend,
)

__all__ = ["MDACSolver", "MdacBackend", "to_unified_backend"]
