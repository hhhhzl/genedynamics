"""
MuJoCo scene adapter for exp3 (closed-loop active perception demo).

Produces an `ObservationBundle` by:
  1. Loading a MuJoCo XML.
  2. Instantiating candidate camera poses (loaded from a JSON).
  3. Rendering each view to an RGB image via MuJoCo's offscreen renderer.

The adapter treats the candidate camera poses as the "frame list"; splits work
the same way as other adapters (train / val / test are contiguous fractions).

Candidate view file format (`candidate_views.json`):
  [
    {"name": "v0", "pos": [0.5, 0.0, 0.3], "quat_wxyz": [1, 0, 0, 0]},
    ...
  ]
  or equivalently a list of 4x4 c2w matrices.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional

import numpy as np

from .base import SceneDataset, SceneDatasetAdapter, SceneDatasetConfig
from .camera_utils import c2w_to_pose7, pose_to_c2w_matrix
from .image_preprocess import (
    apply_exposure_drift,
    composite_rgba,
    normalize_to_01,
    resize_image,
)
from .perturbations import apply_shared_pose_bias, build_shared_pose_bias_se3


@dataclass
class MuJoCoSceneConfig(SceneDatasetConfig):
    """MuJoCo scene config."""

    xml_path: str = ""
    candidate_views_path: str = ""
    camera_name: str = "render_cam"

    # Rendering defaults
    render_height: int = 256
    render_width: int = 256
    render_fov_deg: float = 60.0
    # Fraction of candidate views held out for test.
    train_frac: float = 0.8


@dataclass
class MuJoCoSceneDataset(SceneDataset):
    """Alias."""


def _load_candidate_poses(path: Path) -> np.ndarray:
    """Return (N, 7) poses [px,py,pz, qw,qx,qy,qz]."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        out = []
        for item in data:
            if "pose_7d" in item:
                out.append(np.asarray(item["pose_7d"], dtype=np.float32))
            elif "pos" in item and "quat_wxyz" in item:
                p = np.asarray(item["pos"], dtype=np.float32)
                q = np.asarray(item["quat_wxyz"], dtype=np.float32)
                out.append(np.concatenate([p, q]))
            elif "c2w" in item:
                c2w = np.asarray(item["c2w"], dtype=np.float32).reshape(4, 4)
                out.append(c2w_to_pose7(c2w))
            else:
                raise ValueError(f"Unrecognized candidate view entry: {item}")
        return np.stack(out, axis=0)
    # List of 4x4 matrices.
    c2ws = np.asarray(data, dtype=np.float32)
    if c2ws.ndim == 3 and c2ws.shape[1:] == (4, 4):
        return np.stack([c2w_to_pose7(c) for c in c2ws], axis=0)
    raise ValueError(f"Unrecognized candidate_views format in {path}")


def _fov_to_intrinsics(fov_deg: float, h: int, w: int) -> np.ndarray:
    fov_rad = float(fov_deg) * np.pi / 180.0
    fy = 0.5 * h / np.tan(0.5 * fov_rad)
    fx = fy * (w / h)
    return np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], dtype=np.float32)


def _render_rgb(
    model, data, renderer, camera_name: str, c2w: np.ndarray
) -> np.ndarray:
    """Render a single view given a c2w pose (OpenCV convention)."""
    cam_id = model.camera(camera_name).id
    # MuJoCo uses +X forward, +Z up (body frame). We treat the candidate c2w
    # as giving camera pose in world; set the free-jointed camera body pose.
    # We rely on the XML declaring `mocap="true"` on the camera body and
    # directly writing model's cam_pos / cam_xmat for the target camera.
    pos = c2w[:3, 3]
    R = c2w[:3, :3]
    # MuJoCo camera looks along -Z in its local frame (OpenGL). Convert from
    # OpenCV (looks along +Z) by flipping y and z axes.
    flip = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    R_mjx = R @ flip
    data.cam_xpos[cam_id] = pos
    data.cam_xmat[cam_id] = R_mjx.reshape(-1)
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()
    return rgb  # uint8 (H, W, 3)


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
    raise ValueError(f"Unknown split {split!r}")


class MuJoCoSceneDataAdapter(SceneDatasetAdapter):
    """Renders candidate views from a MuJoCo XML scene."""

    def __init__(self, config: MuJoCoSceneConfig):
        self.config = config
        if not config.xml_path:
            raise ValueError("MuJoCoSceneConfig.xml_path is required")
        if not config.candidate_views_path:
            raise ValueError("MuJoCoSceneConfig.candidate_views_path is required")

        try:
            import mujoco
        except ImportError as e:  # pragma: no cover
            raise ImportError("mujoco is required for MuJoCoSceneDataAdapter") from e
        self._mujoco = mujoco

        xml_path = Path(config.xml_path).expanduser().resolve()
        if not xml_path.exists():
            raise FileNotFoundError(xml_path)
        candidate_path = Path(config.candidate_views_path).expanduser().resolve()
        if not candidate_path.exists():
            raise FileNotFoundError(candidate_path)

        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data = mujoco.MjData(self._model)
        self._candidate_poses = _load_candidate_poses(candidate_path)

    def load_split(
        self, split: Optional[str] = None, resolution: Optional[str] = None
    ) -> MuJoCoSceneDataset:
        mujoco = self._mujoco
        cfg = self.config
        split = split or cfg.split

        if resolution == "eval" and cfg.resolution_eval is not None:
            target_h = target_w = int(cfg.resolution_eval)
        elif resolution == "infer" and cfg.resolution_infer is not None:
            target_h = target_w = int(cfg.resolution_infer)
        else:
            target_h, target_w = cfg.image_height, cfg.image_width

        idx_split = _split_indices(
            self._candidate_poses.shape[0], split, cfg.train_frac
        )
        if cfg.max_views is not None:
            idx_split = idx_split[: cfg.max_views]

        renderer = mujoco.Renderer(
            self._model, height=cfg.render_height, width=cfg.render_width
        )

        apply_stress = split == cfg.perturb_target_split
        drift_mode = cfg.exposure_drift_mode if apply_stress else None
        drift_strength = float(cfg.exposure_drift_strength) if apply_stress else 0.0

        images: List[np.ndarray] = []
        poses_7: List[np.ndarray] = []
        n_views = int(len(idx_split))
        for i, fid in enumerate(idx_split):
            pose = self._candidate_poses[int(fid)]
            c2w = pose_to_c2w_matrix(pose)
            rgb = _render_rgb(self._model, self._data, renderer, cfg.camera_name, c2w)
            rgb = normalize_to_01(rgb)
            rgb = composite_rgba(rgb, cfg.composite_background)
            rgb = resize_image(rgb, target_h, target_w)
            rgb = apply_exposure_drift(
                rgb, view_index=i, n_views=n_views,
                mode=drift_mode, strength=drift_strength,
            )
            images.append(rgb.astype(np.float32))
            poses_7.append(pose)

        images_arr = np.stack(images, axis=0)
        poses_arr = np.stack(poses_7, axis=0)

        if apply_stress and (
            abs(cfg.pose_bias_rotation_deg) > 1e-9
            or abs(cfg.pose_bias_translation_m) > 1e-9
        ):
            delta = build_shared_pose_bias_se3(
                rotation_deg=cfg.pose_bias_rotation_deg,
                translation_m=cfg.pose_bias_translation_m,
                seed=int(cfg.pose_bias_seed),
            )
            poses_arr = apply_shared_pose_bias(poses_arr, delta)

        K = _fov_to_intrinsics(cfg.render_fov_deg, target_h, target_w)
        return MuJoCoSceneDataset(
            images=images_arr,
            camera_poses=poses_arr,
            intrinsics=K,
            frame_paths=[f"mujoco/view_{i}" for i in idx_split.tolist()],
            orig_height=cfg.render_height,
            orig_width=cfg.render_width,
        )


def mujoco_scene_config_from_env_params(
    ep: Mapping[str, Any],
    *,
    adapter_default_split: str,
    image_height: int,
    image_width: int,
) -> MuJoCoSceneConfig:
    return MuJoCoSceneConfig(
        dataset_root=str(ep.get("dataset_root") or ""),
        sequence=ep.get("sequence"),
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
        perturb_target_split=str(ep.get("perturb_target_split", ep.get("split", "train"))),
        pose_bias_rotation_deg=float(ep.get("pose_bias_rotation_deg", 0.0)),
        pose_bias_translation_m=float(ep.get("pose_bias_translation_m", 0.0)),
        pose_bias_seed=int(ep.get("pose_bias_seed", 0)),
        exposure_drift_mode=ep.get("exposure_drift_mode"),
        exposure_drift_strength=float(ep.get("exposure_drift_strength", 0.0)),
        xml_path=str(ep.get("xml_path", "")),
        candidate_views_path=str(ep.get("candidate_views_path", "")),
        camera_name=str(ep.get("camera_name", "render_cam")),
        render_height=int(ep.get("render_height", 256)),
        render_width=int(ep.get("render_width", 256)),
        render_fov_deg=float(ep.get("render_fov_deg", 60.0)),
        train_frac=float(ep.get("train_frac", 0.8)),
    )
