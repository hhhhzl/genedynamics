"""
NeRF Synthetic dataset adapter for MBD3D.

This module loads canonical NeRF Synthetic datasets from
`transforms_{split}.json` and packs them into ObservationBundle.

Supports object-based paths (lego, chair, etc.) and dual resolution
(infer 128x128, eval 512x512).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json

import imageio.v2 as imageio
import numpy as np

from .camera_utils import intrinsics_from_angle_x, transform_matrix_to_pose
from .image_preprocess import prepare_for_inference, prepare_for_eval
from .split import sample_view_indices
from ..types import ObservationBundle


# Canonical NeRF Synthetic object names
NERF_SYNTHETIC_OBJECTS = ("lego", "chair", "drums", "ficus", "hotdog", "materials", "ship")


def resolve_dataset_root(root: str, object_name: Optional[str] = None) -> Path:
    """
    Resolve dataset root for NeRF Synthetic.

    - If root points to object dir (e.g. .../lego) with transforms_*.json, use it.
    - Else if root is parent (e.g. .../nerf_synthetic) and object is given, use root/object.
    """
    p = Path(root).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {p}")

    if object_name is not None:
        obj_dir = p / object_name
        if obj_dir.exists() and (obj_dir / "transforms_train.json").exists():
            return obj_dir
        if (p / "transforms_train.json").exists():
            return p
        return obj_dir if obj_dir.exists() else p

    if (p / "transforms_train.json").exists():
        return p
    return p


@dataclass
class NerfSyntheticConfig:
    dataset_root: str
    object: Optional[str] = None
    split: str = "train"
    image_height: int = 128
    image_width: int = 128
    resolution_infer: Optional[int] = None
    resolution_eval: Optional[int] = None
    composite_background: str = "white"
    pose_convention: str = "opencv"
    max_views: Optional[int] = None
    view_stride: int = 1
    shuffle_seed: Optional[int] = None


@dataclass
class NerfSyntheticDataset:
    images: np.ndarray
    camera_poses: np.ndarray
    intrinsics: np.ndarray
    frame_paths: List[str]
    orig_height: int = 0
    orig_width: int = 0

    def to_observation_bundle(self) -> ObservationBundle:
        timestamps = np.arange(self.images.shape[0], dtype=np.float32)
        return ObservationBundle(
            images=self.images,
            camera_poses=self.camera_poses,
            intrinsics=self.intrinsics,
            timestamps=timestamps,
        )


def _safe_image_path(root: Path, frame_file_path: str) -> Path:
    p = frame_file_path.replace("\\", "/")
    p = p.lstrip("./")
    base = root / p
    if base.suffix:
        if base.exists():
            return base
    else:
        for suffix in (".png", ".jpg", ".jpeg"):
            c = base.with_suffix(suffix)
            if c.exists():
                return c
    for suffix in (".png", ".jpg", ".jpeg"):
        c = root / f"{p}{suffix}"
        if c.exists():
            return c
    raise FileNotFoundError(f"Unable to resolve frame path: {frame_file_path}")


class NerfSyntheticDataAdapter:
    """Industrial-grade loader for NeRF Synthetic to ObservationBundle."""

    def __init__(self, config: NerfSyntheticConfig):
        self.config = config
        self.dataset_root = resolve_dataset_root(
            config.dataset_root,
            object_name=config.object,
        )
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"dataset_root does not exist: {self.dataset_root}")

    def _transforms_path(self, split: str) -> Path:
        return self.dataset_root / f"transforms_{split}.json"

    def load_split(
        self,
        split: Optional[str] = None,
        resolution: Optional[str] = None,
    ) -> NerfSyntheticDataset:
        """
        Load split (train/val/test) with optional resolution override.

        resolution: "infer" | "eval" | None (use config image_height/width)
        """
        split_name = split or self.config.split
        tf_path = self._transforms_path(split_name)
        if not tf_path.exists():
            raise FileNotFoundError(f"Missing NeRF Synthetic metadata: {tf_path}")

        with tf_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        frames = meta.get("frames", [])
        if not frames:
            raise ValueError(f"No frames found in {tf_path}")

        indices = sample_view_indices(
            n_frames=len(frames),
            stride=self.config.view_stride,
            max_views=self.config.max_views,
            shuffle_seed=self.config.shuffle_seed,
        )

        res_infer = self.config.resolution_infer or self.config.image_height
        res_eval = self.config.resolution_eval or self.config.image_width
        if resolution == "infer":
            h, w = res_infer, res_infer
        elif resolution == "eval":
            h, w = res_eval, res_eval
        else:
            h, w = self.config.image_height, self.config.image_width

        images = []
        poses = []
        intrinsics = []
        frame_paths = []

        angle_x = meta.get("camera_angle_x")
        if angle_x is None:
            raise ValueError(f"camera_angle_x missing in {tf_path}")

        for idx in indices:
            frame = frames[int(idx)]
            frame_path = frame.get("file_path")
            transform = frame.get("transform_matrix")
            if frame_path is None or transform is None:
                continue

            img_path = _safe_image_path(self.dataset_root, frame_path)
            img = imageio.imread(img_path)
            img = np.asarray(img, dtype=np.float32)

            img = prepare_for_inference(
                img,
                infer_height=h,
                infer_width=w,
                composite_background=self.config.composite_background,
            )

            pose = transform_matrix_to_pose(
                np.asarray(transform, dtype=np.float32),
                pose_convention=self.config.pose_convention,
            )
            K = intrinsics_from_angle_x(angle_x, h, w)

            images.append(img.astype(np.float32))
            poses.append(pose.astype(np.float32))
            intrinsics.append(K)
            frame_paths.append(str(img_path))

        if not images:
            raise ValueError(f"No valid frames selected from {tf_path}")

        orig_h, orig_w = 800, 800
        if frames and indices.size > 0:
            fr = frames[int(indices[0])]
            fp = fr.get("file_path")
            if fp:
                try:
                    first_img = imageio.imread(_safe_image_path(self.dataset_root, fp))
                    orig_h, orig_w = first_img.shape[:2]
                except Exception:
                    pass

        return NerfSyntheticDataset(
            images=np.stack(images, axis=0).astype(np.float32),
            camera_poses=np.stack(poses, axis=0).astype(np.float32),
            intrinsics=np.stack(intrinsics, axis=0).astype(np.float32),
            frame_paths=frame_paths,
            orig_height=orig_h,
            orig_width=orig_w,
        )

    def load_bundle(self, split: Optional[str] = None) -> ObservationBundle:
        return self.load_split(split=split).to_observation_bundle()

    def load_bundle_dual_resolution(
        self,
        split: Optional[str] = None,
    ) -> tuple[ObservationBundle, ObservationBundle]:
        """Return (infer_bundle, eval_bundle) for dual-resolution protocol."""
        infer_ds = self.load_split(split=split, resolution="infer")
        eval_ds = self.load_split(split=split, resolution="eval")
        return infer_ds.to_observation_bundle(), eval_ds.to_observation_bundle()


def load_nerf_synthetic_bundle(
    dataset_root: str,
    object_name: Optional[str] = None,
    split: str = "train",
    image_height: int = 128,
    image_width: int = 128,
    resolution_infer: Optional[int] = None,
    resolution_eval: Optional[int] = None,
    composite_background: str = "white",
    pose_convention: str = "opencv",
    max_views: Optional[int] = None,
    view_stride: int = 1,
    shuffle_seed: Optional[int] = None,
) -> ObservationBundle:
    """Load NeRF Synthetic as ObservationBundle."""
    adapter = NerfSyntheticDataAdapter(
        NerfSyntheticConfig(
            dataset_root=dataset_root,
            object=object_name,
            split=split,
            image_height=image_height,
            image_width=image_width,
            resolution_infer=resolution_infer,
            resolution_eval=resolution_eval,
            composite_background=composite_background,
            pose_convention=pose_convention,
            max_views=max_views,
            view_stride=view_stride,
            shuffle_seed=shuffle_seed,
        )
    )
    return adapter.load_bundle(split=split)
