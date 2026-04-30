"""
Shared data types for scene observations (dataset-agnostic).

Used by all data adapters (NeRF Synthetic, Replica, TUM, MuJoCo, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

Array = Union[np.ndarray, Any]


@dataclass
class CameraPose:
    """
    Camera pose (extrinsics).

    Attributes:
        position: (3,) camera position in world
        quaternion: (4,) xyzw quaternion for orientation
        rotation_matrix: (3, 3) alternative to quaternion
    """

    position: Array
    quaternion: Optional[Array] = None
    rotation_matrix: Optional[Array] = None

    def to_matrix(self) -> Array:
        """4x4 extrinsic matrix."""
        raise NotImplementedError("Subclass or implement for backend")


@dataclass
class ObservationBundle:
    """
    Bundle of observations for likelihood computation / scene reconstruction.

    Attributes:
        images: list of (H, W, C) images or stacked (N, H, W, C)
        camera_poses: list of CameraPose or (N, 7) [px, py, pz, qw, qx, qy, qz]
        intrinsics: (3, 3) or (N, 3, 3) camera intrinsics
        masks: optional (N, H, W) validity masks
        timestamps: optional (N,) for temporal ordering
        initial_scene: optional pre-trained scene for warm start
    """

    images: Array
    camera_poses: Array
    intrinsics: Optional[Array] = None
    masks: Optional[Array] = None
    timestamps: Optional[Array] = None
    initial_scene: Optional[Any] = None

    @property
    def num_views(self) -> int:
        if hasattr(self.images, "shape"):
            return int(self.images.shape[0]) if self.images.ndim >= 4 else 1
        return len(self.images)
