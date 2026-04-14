"""
Shared data adapters for scene reconstruction experiments.

Currently implemented:
  - NeRF Synthetic (lego / chair / drums / ...)

Planned:
  - Replica (indoor RGB-D, exp2)
  - TUM RGB-D (exp2)
  - MuJoCo synthetic (exp3)

All adapters return `ObservationBundle` (images + camera poses + intrinsics),
which is the input contract for MBD3D, gsplat, and any downstream solver.
"""

from .base import SceneDataset, SceneDatasetAdapter, SceneDatasetConfig
from .types import CameraPose, ObservationBundle
from .camera_utils import (
    intrinsics_from_angle_x,
    focal_from_angle_x,
    transform_matrix_to_pose,
    pose_to_c2w_matrix,
    c2w_to_pose7,
    resize_intrinsics,
)
from .image_preprocess import (
    resize_image,
    composite_rgba,
    prepare_for_inference,
    prepare_for_eval,
    apply_exposure_drift,
)
from .perturbations import build_shared_pose_bias_se3, apply_shared_pose_bias
from .split import sample_view_indices, split_indices
from .nerf_synthetic import (
    NerfSyntheticDataAdapter,
    NerfSyntheticConfig,
    NerfSyntheticDataset,
    load_nerf_synthetic_bundle,
    nerf_synthetic_config_from_env_params,
    resolve_dataset_root,
    NERF_SYNTHETIC_OBJECTS,
)
from .replica import (
    ReplicaDataAdapter,
    ReplicaConfig,
    ReplicaDataset,
    replica_config_from_env_params,
    REPLICA_SEQUENCES,
)
from .tum_rgbd import (
    TUMDataAdapter,
    TUMConfig,
    TUMDataset,
    tum_config_from_env_params,
    TUM_INTRINSICS,
)
from .mujoco_scene import (
    MuJoCoSceneDataAdapter,
    MuJoCoSceneConfig,
    MuJoCoSceneDataset,
    mujoco_scene_config_from_env_params,
)

__all__ = [
    # Base interfaces
    "SceneDataset",
    "SceneDatasetAdapter",
    "SceneDatasetConfig",
    # Shared types
    "CameraPose",
    "ObservationBundle",
    # Camera / intrinsics utils
    "intrinsics_from_angle_x",
    "focal_from_angle_x",
    "transform_matrix_to_pose",
    "pose_to_c2w_matrix",
    "c2w_to_pose7",
    "resize_intrinsics",
    # Image preprocessing
    "resize_image",
    "composite_rgba",
    "prepare_for_inference",
    "prepare_for_eval",
    "apply_exposure_drift",
    # Stress-test perturbations
    "build_shared_pose_bias_se3",
    "apply_shared_pose_bias",
    # View splits
    "sample_view_indices",
    "split_indices",
    # NeRF Synthetic
    "NerfSyntheticDataAdapter",
    "NerfSyntheticConfig",
    "NerfSyntheticDataset",
    "load_nerf_synthetic_bundle",
    "nerf_synthetic_config_from_env_params",
    "resolve_dataset_root",
    "NERF_SYNTHETIC_OBJECTS",
    # Replica
    "ReplicaDataAdapter",
    "ReplicaConfig",
    "ReplicaDataset",
    "replica_config_from_env_params",
    "REPLICA_SEQUENCES",
    # TUM RGB-D
    "TUMDataAdapter",
    "TUMConfig",
    "TUMDataset",
    "tum_config_from_env_params",
    "TUM_INTRINSICS",
    # MuJoCo
    "MuJoCoSceneDataAdapter",
    "MuJoCoSceneConfig",
    "MuJoCoSceneDataset",
    "mujoco_scene_config_from_env_params",
]
