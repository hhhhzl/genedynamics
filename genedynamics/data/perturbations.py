"""
Shared observation perturbations for Experiment 1 stress tests.

Pose bias: one fixed SE(3) Δ applied to every camera (left-multiply on c2w),
            T'_i = Δ @ T_i — models extrinsic calibration bias.

Exposure drift: per-view affine on RGB in [0, 1] after preprocessing — see image_preprocess.
"""

from __future__ import annotations

import numpy as np

from .camera_utils import c2w_to_pose7, pose_to_c2w_matrix


def _rotation_matrix_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues: unit axis, angle in radians -> 3x3 R."""
    u = np.asarray(axis, dtype=np.float64).ravel()
    n = np.linalg.norm(u)
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    u = (u / n).astype(np.float64)
    x, y, z = float(u[0]), float(u[1]), float(u[2])
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1.0 - c
    R = np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float32,
    )
    return R


def build_shared_pose_bias_se3(
    *,
    rotation_deg: float,
    translation_m: float,
    seed: int,
) -> np.ndarray:
    """
    Build 4x4 SE(3) Δ used for all views (reproducible from seed).

    Rotation magnitude is rotation_deg around a seed-stable axis.
    Translation has L2 norm translation_m in a seed-stable direction.
    """
    rot = max(0.0, float(rotation_deg))
    trans = max(0.0, float(translation_m))
    if rot <= 1e-12 and trans <= 1e-12:
        return np.eye(4, dtype=np.float32)

    rng = np.random.default_rng(int(seed))
    axis = rng.standard_normal(3).astype(np.float64)
    angle_rad = rot * (np.pi / 180.0)
    R = _rotation_matrix_axis_angle(axis, angle_rad)

    tdir = rng.standard_normal(3).astype(np.float64)
    tn = np.linalg.norm(tdir)
    if tn < 1e-12:
        t = np.zeros(3, dtype=np.float32)
    else:
        t = (tdir / tn * trans).astype(np.float32)

    delta = np.eye(4, dtype=np.float32)
    delta[:3, :3] = R
    delta[:3, 3] = t
    return delta


def apply_shared_pose_bias(poses_7d: np.ndarray, delta_4x4: np.ndarray) -> np.ndarray:
    """
    Apply left-multiplied bias to (N, 7) poses: T'_i = Δ @ T_i (c2w).

    Args:
        poses_7d: (N, 7) [px,py,pz, qw,qx,qy,qz]
        delta_4x4: (4, 4)

    Returns:
        (N, 7) biased poses
    """
    poses_7d = np.asarray(poses_7d, dtype=np.float32)
    if poses_7d.ndim != 2 or poses_7d.shape[1] != 7:
        raise ValueError(f"Expected poses (N, 7), got {poses_7d.shape}")
    delta_4x4 = np.asarray(delta_4x4, dtype=np.float32)
    if delta_4x4.shape != (4, 4):
        raise ValueError(f"Expected delta (4, 4), got {delta_4x4.shape}")

    out = np.empty_like(poses_7d)
    for i in range(poses_7d.shape[0]):
        c2w = pose_to_c2w_matrix(poses_7d[i])
        c2w_biased = delta_4x4 @ c2w
        out[i] = c2w_to_pose7(c2w_biased)
    return out
