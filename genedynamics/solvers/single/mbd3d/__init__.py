"""
MBD3D (3DGS robust mapping) solver.

Annealed bridge + MCSA for joint scene and camera trajectory estimation.
"""

from .mbd3d import MBD3DSolver
from .types import (
    SceneParams,
    CameraPose,
    ObservationBundle,
    MBD3DResult,
)
from .protocols import (
    SceneRepresentation,
    ObservationRenderer,
    ObservationLikelihood,
    ObservationNoiseModel,
    CameraTrajectoryPrior,
)

__all__ = [
    "MBD3DSolver",
    "SceneParams",
    "CameraPose",
    "ObservationBundle",
    "MBD3DResult",
    "SceneRepresentation",
    "ObservationRenderer",
    "ObservationLikelihood",
    "ObservationNoiseModel",
    "CameraTrajectoryPrior",
]
