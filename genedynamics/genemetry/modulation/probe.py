"""
Probe-blending task modulator.

Blends a probe-derived geometry correction into the base constraint
geometry using a linear combination.  The probe geometry typically
comes from a mini-batch CFS / stepping retraction pass and captures
local feasibility information not present in the mean SDF normals.

The blending formula is::

    output = raw_geometry + alpha * (probe_geometry * window_mask[:, None])

This is backend-agnostic: it works with any array type that supports
elementwise ``+``, ``*``, and ``[:, None]`` broadcasting (NumPy, JAX,
PyTorch).
"""

from typing import Any, Optional

from genedynamics.genemetry.base import TaskModulator


class ProbeModulator(TaskModulator):
    """Blend probe geometry into base constraint geometry."""

    def modulate(
        self,
        raw_geometry: Any,
        probe_geometry: Optional[Any] = None,
        window_mask: Optional[Any] = None,
        alpha: float = 0.0,
    ) -> Any:
        if probe_geometry is None or alpha == 0.0:
            return raw_geometry
        return raw_geometry + alpha * (probe_geometry * window_mask[:, None])
