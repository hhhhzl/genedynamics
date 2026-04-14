"""
Replica dataset adapter (posed RGB-D indoor scenes).

Expected on-disk layout (the standard "replica_v1" / "Nice-SLAM / iMAP" release):

    <dataset_root>/
      <sequence>/                                # e.g. room_0, office_0
        results/
          frame000000.jpg
          depth000000.png
          ...
        traj.txt                                 # per-frame 4x4 pose (row-major, c2w)
        <sequence>_mesh.ply                      # optional GT mesh
    cam_params.json                              # fx, fy, cx, cy, H, W

If your snapshot uses a slightly different layout, adjust `ReplicaConfig.rgb_dir`,
`ReplicaConfig.depth_dir`, `ReplicaConfig.pose_file`, `ReplicaConfig.cam_params_file`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional

import imageio.v2 as imageio
import numpy as np

from .base import SceneDataset, SceneDatasetAdapter, SceneDatasetConfig
from .camera_utils import c2w_to_pose7
from .image_preprocess import (
    apply_exposure_drift,
    composite_rgba,
    normalize_to_01,
    prepare_for_eval,
    resize_image,
)
from .perturbations import apply_shared_pose_bias, build_shared_pose_bias_se3
from .split import sample_view_indices


# Sequences commonly bundled in the Replica release.
REPLICA_SEQUENCES = (
    "room_0", "room_1", "room_2",
    "office_0", "office_1", "office_2", "office_3", "office_4",
)

# Replica RGB values look like natural "holdout" images when we leave aside the
# last 10%. Adapters that need an explicit split should override the fractions
# in config.
_DEFAULT_TRAIN_FRAC = 0.9


@dataclass
class ReplicaConfig(SceneDatasetConfig):
    """Replica-specific config."""

    # Where files live inside a sequence directory (relative paths).
    rgb_dir: str = "results"
    depth_dir: str = "results"
    rgb_pattern: str = "frame{:06d}.jpg"
    depth_pattern: str = "depth{:06d}.png"
    pose_file: str = "traj.txt"
    cam_params_file: str = "cam_params.json"

    # Depth (mm in PNG) -> meters scale.
    depth_scale: float = 1000.0
    # Include depth in the loaded dataset (returned alongside RGB).
    load_depth: bool = False
    # Fraction of frames reserved for train (rest split evenly between val/test).
    train_frac: float = _DEFAULT_TRAIN_FRAC


@dataclass
class ReplicaDataset(SceneDataset):
    """Alias for compatibility with prior per-adapter Dataset names."""


def _resolve_sequence_root(root: Path, sequence: Optional[str]) -> Path:
    """Locate the sequence directory under `root`. Mirrors NeRF Synthetic logic."""
    if not root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {root}")
    if sequence is not None:
        candidate = root / sequence
        if candidate.exists():
            return candidate
        # Maybe `root` already points at a sequence directory.
        if (root / "traj.txt").exists():
            return root
        raise FileNotFoundError(
            f"Replica sequence {sequence!r} not found under {root}"
        )
    if (root / "traj.txt").exists():
        return root
    raise ValueError("Replica adapter requires a sequence (no traj.txt at root).")


def _load_intrinsics(seq_dir: Path, cfg: ReplicaConfig) -> "tuple[np.ndarray, int, int]":
    """Return (K, H, W) from cam_params.json, falling back to sane defaults."""
    candidates = [seq_dir / cfg.cam_params_file, seq_dir.parent / cfg.cam_params_file]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            cam = data.get("camera", data)
            fx = float(cam["fx"])
            fy = float(cam.get("fy", fx))
            cx = float(cam["cx"])
            cy = float(cam["cy"])
            H = int(cam["h"])
            W = int(cam["w"])
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            return K, H, W
    raise FileNotFoundError(
        f"cam_params.json not found near {seq_dir}; cannot load Replica intrinsics."
    )


def _load_trajectory(pose_path: Path) -> np.ndarray:
    """Parse traj.txt: each line is 12 or 16 floats, 4x4 c2w row-major."""
    arr = np.loadtxt(str(pose_path), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] == 16:
        c2w = arr.reshape(-1, 4, 4)
    elif arr.shape[1] == 12:
        c2w = np.zeros((arr.shape[0], 4, 4), dtype=np.float32)
        c2w[:, :3, :] = arr.reshape(-1, 3, 4)
        c2w[:, 3, 3] = 1.0
    else:
        raise ValueError(f"Unexpected traj.txt column count: {arr.shape[1]}")
    return c2w


def _split_indices(n_total: int, split: str, train_frac: float) -> np.ndarray:
    """Deterministic contiguous split (train | val | test) over frame ids."""
    n_train = int(round(n_total * train_frac))
    n_remain = n_total - n_train
    n_val = n_remain // 2
    n_test = n_remain - n_val
    if split == "train":
        return np.arange(0, n_train, dtype=np.int64)
    if split == "val":
        return np.arange(n_train, n_train + n_val, dtype=np.int64)
    if split == "test":
        return np.arange(n_train + n_val, n_total, dtype=np.int64)
    raise ValueError(f"Unknown split {split!r} (expected train/val/test)")


class ReplicaDataAdapter(SceneDatasetAdapter):
    """Loader for Replica RGB(-D) sequences."""

    def __init__(self, config: ReplicaConfig):
        self.config = config
        self._seq_dir = _resolve_sequence_root(
            Path(config.dataset_root).expanduser().resolve(),
            config.sequence,
        )

    def load_split(
        self,
        split: Optional[str] = None,
        resolution: Optional[str] = None,
    ) -> ReplicaDataset:
        cfg = self.config
        split = split or cfg.split

        # Pick resolution
        if resolution == "eval" and cfg.resolution_eval is not None:
            target_h = target_w = int(cfg.resolution_eval)
        elif resolution == "infer" and cfg.resolution_infer is not None:
            target_h = target_w = int(cfg.resolution_infer)
        else:
            target_h, target_w = cfg.image_height, cfg.image_width

        K_orig, orig_h, orig_w = _load_intrinsics(self._seq_dir, cfg)
        c2w_all = _load_trajectory(self._seq_dir / cfg.pose_file)
        n_total = c2w_all.shape[0]

        split_ids = _split_indices(n_total, split, cfg.train_frac)
        sampled = sample_view_indices(
            len(split_ids),
            max_views=cfg.max_views,
            view_stride=cfg.view_stride,
            shuffle_seed=cfg.shuffle_seed,
        )
        frame_ids = split_ids[sampled]

        rgb_dir = self._seq_dir / cfg.rgb_dir
        depth_dir = self._seq_dir / cfg.depth_dir

        # Adjust intrinsics for the loaded resolution
        sx = target_w / orig_w
        sy = target_h / orig_h
        K = K_orig.copy()
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy

        images: List[np.ndarray] = []
        depths: List[np.ndarray] = []
        frame_paths: List[str] = []
        poses: List[np.ndarray] = []
        n_views = int(len(frame_ids))

        apply_stress = split == cfg.perturb_target_split
        drift_mode = cfg.exposure_drift_mode if apply_stress else None
        drift_strength = float(cfg.exposure_drift_strength) if apply_stress else 0.0

        for view_index, fid in enumerate(frame_ids):
            rgb_path = rgb_dir / cfg.rgb_pattern.format(int(fid))
            if not rgb_path.exists():
                raise FileNotFoundError(f"Replica RGB not found: {rgb_path}")
            rgb = imageio.imread(rgb_path)
            rgb = normalize_to_01(rgb)
            rgb = composite_rgba(rgb, cfg.composite_background)
            rgb = resize_image(rgb, target_h, target_w)
            rgb = apply_exposure_drift(
                rgb, view_index=view_index, n_views=n_views,
                mode=drift_mode, strength=drift_strength,
            )
            images.append(rgb.astype(np.float32))
            frame_paths.append(str(rgb_path))
            poses.append(c2w_all[int(fid)])

            if cfg.load_depth:
                depth_path = depth_dir / cfg.depth_pattern.format(int(fid))
                if depth_path.exists():
                    dpt = imageio.imread(depth_path).astype(np.float32)
                    dpt = dpt / cfg.depth_scale
                    depths.append(resize_image(dpt[..., None], target_h, target_w)[..., 0])
                else:
                    depths.append(np.zeros((target_h, target_w), dtype=np.float32))

        images_arr = np.stack(images, axis=0)
        poses_c2w = np.stack(poses, axis=0)
        poses_7 = np.stack([c2w_to_pose7(p) for p in poses_c2w], axis=0)

        if apply_stress and (
            abs(cfg.pose_bias_rotation_deg) > 1e-9
            or abs(cfg.pose_bias_translation_m) > 1e-9
        ):
            delta = build_shared_pose_bias_se3(
                rotation_deg=cfg.pose_bias_rotation_deg,
                translation_m=cfg.pose_bias_translation_m,
                seed=int(cfg.pose_bias_seed),
            )
            poses_7 = apply_shared_pose_bias(poses_7, delta)

        depth_arr = np.stack(depths, axis=0) if depths else None
        return ReplicaDataset(
            images=images_arr,
            camera_poses=poses_7,
            intrinsics=K.astype(np.float32),
            frame_paths=frame_paths,
            orig_height=orig_h,
            orig_width=orig_w,
            depth=depth_arr,
        )


def replica_config_from_env_params(
    ep: Mapping[str, Any],
    *,
    adapter_default_split: str,
    image_height: int,
    image_width: int,
    return_view_masks: bool = False,
) -> ReplicaConfig:
    mode = ep.get("exposure_drift_mode")
    if mode is not None and mode not in ("constant_add", "linear_gain"):
        raise ValueError(
            f"exposure_drift_mode must be null, 'constant_add', or 'linear_gain', got {mode!r}"
        )
    pert_split = ep.get("perturb_target_split", ep.get("split", "train"))
    return ReplicaConfig(
        dataset_root=str(ep.get("dataset_root") or ep.get("dataset_path", "")),
        sequence=ep.get("sequence") or ep.get("object"),
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
        load_depth=bool(ep.get("load_depth", False)),
        train_frac=float(ep.get("train_frac", _DEFAULT_TRAIN_FRAC)),
    )
