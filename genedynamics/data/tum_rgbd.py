"""
TUM RGB-D dataset adapter (e.g. fr1/desk, fr1/room, fr2/xyz).

Expected on-disk layout (matches the TUM RGB-D release):

    <dataset_root>/
      rgbd_dataset_freiburg1_desk/
        rgb.txt                                  # "ts rgb_rel_path"
        depth.txt                                # "ts depth_rel_path"
        groundtruth.txt                          # "ts tx ty tz qx qy qz qw"
        rgb/*.png
        depth/*.png
        accelerometer.txt
      rgbd_dataset_freiburg1_xyz/
        ...

RGB and depth frames are indexed by timestamp; we triple-associate
(rgb, depth, gt_pose) by finding the closest-timestamp match within
`max_time_diff` seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

import imageio.v2 as imageio
import numpy as np

from .base import SceneDataset, SceneDatasetAdapter, SceneDatasetConfig
from .image_preprocess import (
    apply_exposure_drift,
    composite_rgba,
    normalize_to_01,
    resize_image,
)
from .perturbations import apply_shared_pose_bias, build_shared_pose_bias_se3
from .split import sample_view_indices


# TUM default intrinsics per Freiburg setup (ros param from tum_rgbd manual).
TUM_INTRINSICS = {
    "freiburg1": {"fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3, "w": 640, "h": 480},
    "freiburg2": {"fx": 520.9, "fy": 521.0, "cx": 325.1, "cy": 249.7, "w": 640, "h": 480},
    "freiburg3": {"fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6, "w": 640, "h": 480},
}


@dataclass
class TUMConfig(SceneDatasetConfig):
    """TUM RGB-D specific config."""

    # Filenames (relative to sequence dir).
    rgb_list_file: str = "rgb.txt"
    depth_list_file: str = "depth.txt"
    groundtruth_file: str = "groundtruth.txt"

    # Association tolerance (seconds).
    max_time_diff: float = 0.02
    # Depth PNG -> meters scale.
    depth_scale: float = 5000.0
    # Include depth in the loaded dataset.
    load_depth: bool = False
    # Hold out tail fraction for val/test.
    train_frac: float = 0.9


@dataclass
class TUMDataset(SceneDataset):
    """Alias for compatibility."""


def _parse_tum_list(path: Path) -> np.ndarray:
    """Parse a TUM list file into (N, ) stamps and (N, ) paths (object dtype)."""
    stamps: List[float] = []
    paths: List[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            stamps.append(float(parts[0]))
            paths.append(parts[1])
    return np.array(stamps, dtype=np.float64), np.array(paths, dtype=object)


def _parse_groundtruth(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse groundtruth.txt -> (stamps, poses_7d) where pose_7d is [x y z qw qx qy qz]."""
    stamps: List[float] = []
    poses: List[np.ndarray] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            stamps.append(ts)
            poses.append(np.array([tx, ty, tz, qw, qx, qy, qz], dtype=np.float32))
    return np.array(stamps, dtype=np.float64), np.stack(poses, axis=0)


def _associate(
    rgb_ts: np.ndarray, other_ts: np.ndarray, max_diff: float
) -> np.ndarray:
    """For each rgb_ts, return the index into other_ts of the closest stamp, or -1."""
    # other_ts sorted -> searchsorted
    order = np.argsort(other_ts)
    ot_sorted = other_ts[order]
    idxs = np.searchsorted(ot_sorted, rgb_ts)
    out = np.full(rgb_ts.shape[0], -1, dtype=np.int64)
    for i, ts in enumerate(rgb_ts):
        candidates = []
        for k in (idxs[i] - 1, idxs[i]):
            if 0 <= k < ot_sorted.shape[0]:
                candidates.append((abs(ot_sorted[k] - ts), order[k]))
        if not candidates:
            continue
        dt, pick = min(candidates, key=lambda x: x[0])
        if dt <= max_diff:
            out[i] = pick
    return out


def _fr_prefix(sequence: str) -> str:
    """Extract freiburg1/2/3 from a TUM sequence name."""
    s = sequence.lower()
    for key in ("freiburg1", "freiburg2", "freiburg3"):
        if key in s:
            return key
    raise ValueError(f"Cannot infer freiburg-N from sequence: {sequence!r}")


def _split_indices(n_total: int, split: str, train_frac: float) -> np.ndarray:
    n_train = int(round(n_total * train_frac))
    n_remain = n_total - n_train
    n_val = n_remain // 2
    if split == "train":
        return np.arange(0, n_train, dtype=np.int64)
    if split == "val":
        return np.arange(n_train, n_train + n_val, dtype=np.int64)
    if split == "test":
        return np.arange(n_train + n_val, n_total, dtype=np.int64)
    raise ValueError(f"Unknown split {split!r} (expected train/val/test)")


class TUMDataAdapter(SceneDatasetAdapter):
    """Loader for TUM RGB-D sequences."""

    def __init__(self, config: TUMConfig):
        self.config = config
        root = Path(config.dataset_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"dataset_root does not exist: {root}")
        seq = config.sequence
        if seq is None:
            raise ValueError("TUM adapter requires `sequence` (e.g. 'rgbd_dataset_freiburg1_desk').")
        seq_dir = root / seq
        if not seq_dir.exists() and (root / "rgb.txt").exists():
            # Root already points at the sequence directory.
            seq_dir = root
        if not seq_dir.exists():
            raise FileNotFoundError(f"TUM sequence not found: {seq_dir}")
        self._seq_dir = seq_dir

    def load_split(
        self, split: Optional[str] = None, resolution: Optional[str] = None
    ) -> TUMDataset:
        cfg = self.config
        split = split or cfg.split
        if resolution == "eval" and cfg.resolution_eval is not None:
            target_h = target_w = int(cfg.resolution_eval)
        elif resolution == "infer" and cfg.resolution_infer is not None:
            target_h = target_w = int(cfg.resolution_infer)
        else:
            target_h, target_w = cfg.image_height, cfg.image_width

        rgb_ts, rgb_rel = _parse_tum_list(self._seq_dir / cfg.rgb_list_file)
        depth_ts, depth_rel = _parse_tum_list(self._seq_dir / cfg.depth_list_file)
        gt_ts, gt_pose = _parse_groundtruth(self._seq_dir / cfg.groundtruth_file)

        depth_idx = _associate(rgb_ts, depth_ts, cfg.max_time_diff)
        gt_idx = _associate(rgb_ts, gt_ts, cfg.max_time_diff)
        valid = (gt_idx >= 0) & (depth_idx >= 0 if cfg.load_depth else True)
        valid_ids = np.where(valid)[0]

        split_valid = valid_ids[_split_indices(len(valid_ids), split, cfg.train_frac)]
        sampled = sample_view_indices(
            len(split_valid),
            max_views=cfg.max_views,
            view_stride=cfg.view_stride,
            shuffle_seed=cfg.shuffle_seed,
        )
        frame_ids = split_valid[sampled]

        # Intrinsics
        fr = _fr_prefix(cfg.sequence)
        k = TUM_INTRINSICS[fr]
        orig_h, orig_w = k["h"], k["w"]
        fx, fy, cx, cy = k["fx"], k["fy"], k["cx"], k["cy"]
        sx = target_w / orig_w
        sy = target_h / orig_h
        K = np.array(
            [[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], dtype=np.float32
        )

        apply_stress = split == cfg.perturb_target_split
        drift_mode = cfg.exposure_drift_mode if apply_stress else None
        drift_strength = float(cfg.exposure_drift_strength) if apply_stress else 0.0

        images: List[np.ndarray] = []
        depths: List[np.ndarray] = []
        poses: List[np.ndarray] = []
        stamps: List[float] = []
        frame_paths: List[str] = []
        n_views = int(len(frame_ids))

        for view_index, fid in enumerate(frame_ids):
            rgb_path = self._seq_dir / rgb_rel[fid]
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
            poses.append(gt_pose[gt_idx[fid]])
            stamps.append(float(rgb_ts[fid]))

            if cfg.load_depth and depth_idx[fid] >= 0:
                dpath = self._seq_dir / depth_rel[depth_idx[fid]]
                dpt = imageio.imread(dpath).astype(np.float32) / cfg.depth_scale
                depths.append(resize_image(dpt[..., None], target_h, target_w)[..., 0])

        images_arr = np.stack(images, axis=0)
        poses_7 = np.stack(poses, axis=0)

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
        return TUMDataset(
            images=images_arr,
            camera_poses=poses_7,
            intrinsics=K.astype(np.float32),
            frame_paths=frame_paths,
            orig_height=orig_h,
            orig_width=orig_w,
            depth=depth_arr,
            timestamps=np.array(stamps, dtype=np.float32),
        )


def tum_config_from_env_params(
    ep: Mapping[str, Any],
    *,
    adapter_default_split: str,
    image_height: int,
    image_width: int,
    return_view_masks: bool = False,
) -> TUMConfig:
    mode = ep.get("exposure_drift_mode")
    if mode is not None and mode not in ("constant_add", "linear_gain"):
        raise ValueError(
            f"exposure_drift_mode must be null, 'constant_add', or 'linear_gain', got {mode!r}"
        )
    pert_split = ep.get("perturb_target_split", ep.get("split", "train"))
    return TUMConfig(
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
        max_time_diff=float(ep.get("max_time_diff", 0.02)),
        train_frac=float(ep.get("train_frac", 0.9)),
    )
