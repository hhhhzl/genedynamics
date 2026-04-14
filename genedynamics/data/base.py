"""
Base interfaces for scene-reconstruction dataset adapters.

Every dataset adapter (NeRF Synthetic, Replica, TUM RGB-D, MuJoCo, ...) must:
  - Subclass `SceneDatasetConfig` for YAML-friendly configuration.
  - Return a `SceneDataset` that converts to `ObservationBundle` via
    `to_observation_bundle()`.
  - Subclass `SceneDatasetAdapter` and implement `load_split(...)`.

Common stress-test perturbations (pose bias, exposure drift) are expressed
via fields on `SceneDatasetConfig` and applied uniformly across adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .types import ObservationBundle


@dataclass
class SceneDatasetConfig:
    """Shared configuration across all scene-reconstruction dataset adapters."""

    # Filesystem / identity
    dataset_root: str
    sequence: Optional[str] = None
    split: str = "train"

    # Image preprocessing
    image_height: int = 128
    image_width: int = 128
    resolution_infer: Optional[int] = None
    resolution_eval: Optional[int] = None
    composite_background: str = "white"
    pose_convention: str = "opencv"

    # View selection
    max_views: Optional[int] = None
    view_stride: int = 1
    shuffle_seed: Optional[int] = None

    # Perturbation guard: only apply stress when loading this split.
    perturb_target_split: str = "train"

    # Stress: shared extrinsic bias
    pose_bias_rotation_deg: float = 0.0
    pose_bias_translation_m: float = 0.0
    pose_bias_seed: int = 0

    # Stress: exposure drift on RGB
    exposure_drift_mode: Optional[str] = None
    exposure_drift_strength: float = 0.0

    # Foreground masks (for mask-weighted loss)
    return_view_masks: bool = False


@dataclass
class SceneDataset:
    """
    Unified container for a loaded split of any scene dataset.

    Fields not provided by a given dataset (e.g. depth for a monocular set)
    should be left as None.
    """

    images: np.ndarray                              # (N, H, W, 3) float32 in [0, 1]
    camera_poses: np.ndarray                        # (N, 7) [px,py,pz, qw,qx,qy,qz]
    intrinsics: np.ndarray                          # (3, 3) or (N, 3, 3)
    frame_paths: List[str] = field(default_factory=list)
    orig_height: int = 0
    orig_width: int = 0
    masks: Optional[np.ndarray] = None              # (N, H, W, 1) foreground alpha
    depth: Optional[np.ndarray] = None              # (N, H, W) meters
    timestamps: Optional[np.ndarray] = None         # (N,) float seconds

    def to_observation_bundle(self) -> ObservationBundle:
        if self.timestamps is not None:
            ts = np.asarray(self.timestamps, dtype=np.float32)
        else:
            ts = np.arange(self.images.shape[0], dtype=np.float32)
        return ObservationBundle(
            images=self.images,
            camera_poses=self.camera_poses,
            intrinsics=self.intrinsics,
            masks=self.masks,
            timestamps=ts,
        )


class SceneDatasetAdapter(ABC):
    """Abstract base class for dataset adapters."""

    config: SceneDatasetConfig

    @abstractmethod
    def load_split(
        self, split: Optional[str] = None, resolution: Optional[str] = None
    ) -> SceneDataset:
        """Load a split ("train" | "val" | "test") at optional resolution label."""
        raise NotImplementedError

    def load_bundle(self, split: Optional[str] = None) -> ObservationBundle:
        """Convenience shortcut: load_split(...).to_observation_bundle()."""
        return self.load_split(split=split).to_observation_bundle()
