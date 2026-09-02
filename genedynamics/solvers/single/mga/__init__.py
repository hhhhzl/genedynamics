"""MGA solver — Manifold Generative Annealing (multi-backend).

DIAL's MBD annealing made manifold-aware: a receding-horizon sampling controller
over a lower-control primitive `U=(r,K,nu)`, with soft-feasibility weighting,
constraint-manifold tangent denoising + retraction, an optional model-free RL
prior, and a coupled DDPM/DDIM/flow annealing schedule
(`docs/mga/MGA_README.md`, `idea.txt`).

Composes the config surface, registry entry, the `core/` component math, and
the reverse step over the receding-horizon bridge.
"""

from genedynamics.solvers.single.mga.mga import MGASolver
from genedynamics.solvers.single.mga.backend_impl import (
    MgaBackend,
    to_unified_backend,
)

__all__ = ["MGASolver", "MgaBackend", "to_unified_backend"]
