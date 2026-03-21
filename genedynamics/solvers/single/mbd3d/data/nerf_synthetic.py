"""
NeRF Synthetic dataset adapter for MBD3D.

This module loads canonical NeRF Synthetic datasets from
`transforms_{split}.json` and packs them into ObservationBundle.

Supports object-based paths (lego, chair, etc.) and dual resolution
(infer 128x128, eval 512x512).

Experiment 1 stress tests: shared SE(3) pose bias, exposure drift (see perturbations / image_preprocess).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional
import json

import imageio.v2 as imageio
import numpy as np

from .camera_utils import intrinsics_from_angle_x, transform_matrix_to_pose
from .image_preprocess import (
    apply_exposure_drift,
    composite_rgba,
    normalize_to_01,
    prepare_for_inference,
    resize_image,
)
from .perturbations import apply_shared_pose_bias, build_shared_pose_bias_se3
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
    # Apply pose/exposure stress only when load_split(split=...) matches this (usually "train").
    perturb_target_split: str = "train"
    # Stress test: shared extrinsic bias (degrees + meters); seed fixes Δ across runs.
    pose_bias_rotation_deg: float = 0.0
    pose_bias_translation_m: float = 0.0
    pose_bias_seed: int = 0
    # Stress test: exposure-style drift on RGB after preprocessing (see apply_exposure_drift).
    exposure_drift_mode: Optional[str] = None
    exposure_drift_strength: float = 0.0
    # Foreground alpha masks for gsplat mask-weighted loss (N, H, W, 1).
    return_view_masks: bool = False


@dataclass
class NerfSyntheticDataset:
    images: np.ndarray
    camera_poses: np.ndarray
    intrinsics: np.ndarray
    frame_paths: List[str]
    orig_height: int = 0
    orig_width: int = 0
    masks: Optional[np.ndarray] = None

    def to_observation_bundle(self) -> ObservationBundle:
        timestamps = np.arange(self.images.shape[0], dtype=np.float32)
        return ObservationBundle(
            images=self.images,
            camera_poses=self.camera_poses,
            intrinsics=self.intrinsics,
            timestamps=timestamps,
        )


def nerf_synthetic_config_from_env_params(
    ep: Mapping[str, Any],
    *,
    adapter_default_split: str,
    image_height: int,
    image_width: int,
    return_view_masks: bool = False,
) -> NerfSyntheticConfig:
    """
    Single factory for YAML env_params -> NerfSyntheticConfig (MBD, baselines, stress tests).

    Keeps perturbation and sampling flags aligned across scripts.
    """
    root = ep.get("dataset_root") or ep.get("dataset_path", "")
    mode = ep.get("exposure_drift_mode")
    if mode is not None and mode not in ("constant_add", "linear_gain"):
        raise ValueError(
            f"exposure_drift_mode must be null, 'constant_add', or 'linear_gain', got {mode!r}"
        )
    pert_split = ep.get("perturb_target_split")
    if pert_split is None:
        pert_split = str(ep.get("split", "train"))
    return NerfSyntheticConfig(
        dataset_root=str(root),
        object=ep.get("object"),
        split=str(adapter_default_split),
        image_height=int(image_height),
        image_width=int(image_width),
        resolution_infer=ep.get("resolution_infer"),
        resolution_eval=ep.get("resolution_eval") or ep.get("eval_resolution"),
        composite_background=str(ep.get("composite_background", "white")),
        pose_convention=str(ep.get("pose_convention", "opencv")),
        max_views=ep.get("max_views"),
        view_stride=int(ep.get("view_stride", 1)),
        shuffle_seed=ep.get("shuffle_seed"),
        perturb_target_split=str(pert_split),
        pose_bias_rotation_deg=float(ep.get("pose_bias_rotation_deg", 0.0)),
        pose_bias_translation_m=float(ep.get("pose_bias_translation_m", 0.0)),
        pose_bias_seed=int(ep.get("pose_bias_seed", 0)),
        exposure_drift_mode=mode,
        exposure_drift_strength=float(ep.get("exposure_drift_strength", 0.0)),
        return_view_masks=bool(return_view_masks),
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

        images: List[np.ndarray] = []
        poses: List[np.ndarray] = []
        intrinsics: List[np.ndarray] = []
        frame_paths: List[str] = []
        masks_list: List[np.ndarray] = []

        angle_x = meta.get("camera_angle_x")
        if angle_x is None:
            raise ValueError(f"camera_angle_x missing in {tf_path}")

        n_sel = int(indices.size)
        apply_stress = split_name == self.config.perturb_target_split
        drift_mode = self.config.exposure_drift_mode if apply_stress else None
        drift_strength = float(self.config.exposure_drift_strength) if apply_stress else 0.0

        for local_i, idx in enumerate(indices):
            frame = frames[int(idx)]
            frame_path = frame.get("file_path")
            transform = frame.get("transform_matrix")
            if frame_path is None or transform is None:
                continue

            img_path = _safe_image_path(self.dataset_root, frame_path)
            raw = np.asarray(imageio.imread(img_path), dtype=np.float32)
            raw = normalize_to_01(raw)

            if self.config.return_view_masks:
                if raw.ndim == 3 and raw.shape[-1] >= 4:
                    alpha = raw[..., 3]
                else:
                    alpha = np.ones(raw.shape[:2], dtype=np.float32)
                rgb = composite_rgba(raw, self.config.composite_background)
                rgb = resize_image(rgb, h, w)
                mask_hw1 = resize_image(alpha, h, w)[..., np.newaxis].astype(np.float32)
                masks_list.append(mask_hw1)
            else:
                rgb = prepare_for_inference(
                    raw,
                    infer_height=h,
                    infer_width=w,
                    composite_background=self.config.composite_background,
                )

            rgb = apply_exposure_drift(
                rgb,
                view_index=local_i,
                n_views=n_sel,
                mode=drift_mode,
                strength=drift_strength,
            )

            pose = transform_matrix_to_pose(
                np.asarray(transform, dtype=np.float32),
                pose_convention=self.config.pose_convention,
            )
            K = intrinsics_from_angle_x(angle_x, h, w)

            images.append(rgb.astype(np.float32))
            poses.append(pose.astype(np.float32))
            intrinsics.append(K)
            frame_paths.append(str(img_path))

        if not images:
            raise ValueError(f"No valid frames selected from {tf_path}")

        poses_arr = np.stack(poses, axis=0).astype(np.float32)
        if apply_stress and (
            abs(float(self.config.pose_bias_rotation_deg)) > 1e-9
            or abs(float(self.config.pose_bias_translation_m)) > 1e-9
        ):
            delta = build_shared_pose_bias_se3(
                rotation_deg=self.config.pose_bias_rotation_deg,
                translation_m=self.config.pose_bias_translation_m,
                seed=int(self.config.pose_bias_seed),
            )
            poses_arr = apply_shared_pose_bias(poses_arr, delta)

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

        masks_stacked: Optional[np.ndarray] = None
        if masks_list:
            masks_stacked = np.stack(masks_list, axis=0).astype(np.float32)

        return NerfSyntheticDataset(
            images=np.stack(images, axis=0).astype(np.float32),
            camera_poses=poses_arr,
            intrinsics=np.stack(intrinsics, axis=0).astype(np.float32),
            frame_paths=frame_paths,
            orig_height=orig_h,
            orig_width=orig_w,
            masks=masks_stacked,
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
    perturb_target_split: str = "train",
    pose_bias_rotation_deg: float = 0.0,
    pose_bias_translation_m: float = 0.0,
    pose_bias_seed: int = 0,
    exposure_drift_mode: Optional[str] = None,
    exposure_drift_strength: float = 0.0,
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
            perturb_target_split=perturb_target_split,
            pose_bias_rotation_deg=pose_bias_rotation_deg,
            pose_bias_translation_m=pose_bias_translation_m,
            pose_bias_seed=pose_bias_seed,
            exposure_drift_mode=exposure_drift_mode,
            exposure_drift_strength=exposure_drift_strength,
        )
    )
    return adapter.load_bundle(split=split)
