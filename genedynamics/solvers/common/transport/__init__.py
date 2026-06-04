"""
Swappable reverse-transport package for the MBD-family solvers.

See docs/methods/transport_unification_plan.md. ``ReverseTransport`` is the
shared interface; ``backends/`` holds family implementations (DDPM, DDIM, FM).
Solvers default ``transport=None`` (verbatim inline default).

This package lives under ``genedynamics/solvers/common/`` so the MBD-family and
2GO-family solvers can share the same reverse-transport backends.
"""

from typing import Optional

from genedynamics.solvers.common.transport.base import ReverseTransport
from genedynamics.solvers.common.transport.backends.ddpm_jax import DDPMTransport
from genedynamics.solvers.common.transport.backends.ddim_jax import DDIMTransport
from genedynamics.solvers.common.transport.backends.fm_jax import FMTransport
from genedynamics.solvers.common.transport.backends.adaptive_jax import (
    AdaptiveTransport,
    FAMILY_TO_IDX,
)

#: family tag -> backend class (case-insensitive lookup via ``make_transport``).
_TRANSPORT_REGISTRY = {
    "DDPM": DDPMTransport,
    "DDIM": DDIMTransport,
    "FM": FMTransport,
}


def make_transport(family: Optional[str]) -> Optional[ReverseTransport]:
    """Build the reverse-transport backend for a family tag.

    Args:
        family: One of "DDPM" | "DDIM" | "FM" (case-insensitive). ``None`` or
            "DDPM" returns ``None`` so the solver keeps its byte-identical inline
            DDPM path (a ``DDPMTransport`` instance reproduces it exactly, but the
            ``None`` sentinel is the verbatim default contract).

    Returns:
        A ``ReverseTransport`` for DDIM/FM, or ``None`` for DDPM/None.
    """
    if family is None:
        return None
    fam = str(family).upper()
    if fam == "DDPM":
        return None
    if fam not in _TRANSPORT_REGISTRY:
        raise ValueError(
            f"Unknown transport family {family!r}; expected one of "
            f"{tuple(_TRANSPORT_REGISTRY)}."
        )
    return _TRANSPORT_REGISTRY[fam]()


__all__ = [
    "ReverseTransport",
    "DDPMTransport",
    "DDIMTransport",
    "FMTransport",
    "AdaptiveTransport",
    "FAMILY_TO_IDX",
    "make_transport",
]
