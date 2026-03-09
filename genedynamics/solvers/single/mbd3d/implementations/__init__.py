"""
Reference implementations for MBD3D protocols.

- GaussianSplatScene: simple 3DGS scene representation
- MockRenderer: placeholder renderer for testing (replace with diff-gaussian-splatting)
- GaussianObservationLikelihood: L2 / low-rank noise likelihood
"""

from .scene import GaussianSplatScene
from .renderer import MockRenderer
from .likelihood import GaussianObservationLikelihood

__all__ = [
    "GaussianSplatScene",
    "MockRenderer",
    "GaussianObservationLikelihood",
]
