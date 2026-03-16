"""
Camera utilities for MBD3D / 3DGS data adapters.

Handles intrinsics from NeRF-style camera_angle_x, coordinate conventions
(OpenGL/OpenCV), and pose conversions.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def focal_from_angle_x(angle_x: float, width: int) -> float:
    """
    Compute focal length from horizontal field-of-view angle.

    f = 0.5 * W / tan(0.5 * angle_x)
    """
    return 0.5 * float(width) / np.tan(0.5 * float(angle_x))


def intrinsics_from_angle_x(
    angle_x: float,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Build 3x3 intrinsics matrix from camera_angle_x and resolution.

    Uses square pixels (fx = fy).
    """
    fx = focal_from_angle_x(angle_x, width)
    fy = fx
    cx = 0.5 * float(width)
    cy = 0.5 * float(height)
    K = np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return K


def opengl_to_opencv_transform() -> np.ndarray:
    """
    4x4 matrix to convert OpenGL camera convention to OpenCV.

    OpenGL: Y up, Z backward (into screen)
    OpenCV: Y down, Z forward
    """
    return np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def transform_matrix_to_pose(
    transform_matrix: np.ndarray,
    pose_convention: str = "opencv",
) -> np.ndarray:
    """
    Extract 7D pose (position + quaternion wxyz) from 4x4 c2w matrix.

    Args:
        transform_matrix: (4, 4) camera-to-world
        pose_convention: "opencv" applies OpenGL->OpenCV before extraction

    Returns:
        (7,) [px, py, pz, qw, qx, qy, qz]
    """
    c2w = np.asarray(transform_matrix, dtype=np.float32)
    if c2w.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got {c2w.shape}")

    if pose_convention == "opencv":
        c2w = c2w @ opengl_to_opencv_transform()

    t = c2w[:3, 3]
    R = c2w[:3, :3]
    q = rotation_matrix_to_quaternion_wxyz(R)
    return np.concatenate([t.astype(np.float32), q.astype(np.float32)], axis=0)


def rotation_matrix_to_quaternion_wxyz(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion (w, x, y, z).
    """
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.asarray([qw, qx, qy, qz], dtype=np.float32)
    n = np.linalg.norm(q)
    return q / (n + 1e-8)


def quaternion_wxyz_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float32).ravel()
    if q.size != 4:
        raise ValueError(f"Expected 4 elements, got {q.size}")
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def pose_to_c2w_matrix(pose: np.ndarray) -> np.ndarray:
    """
    Build 4x4 c2w from 7D pose [px, py, pz, qw, qx, qy, qz].
    """
    pose = np.asarray(pose, dtype=np.float32).ravel()
    if pose.size != 7:
        raise ValueError(f"Expected 7D pose, got {pose.size}")
    t = pose[:3]
    q = pose[3:7]
    R = quaternion_wxyz_to_rotation_matrix(q)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R
    c2w[:3, 3] = t
    return c2w


def resize_intrinsics(
    K: np.ndarray,
    orig_height: int,
    orig_width: int,
    new_height: int,
    new_width: int,
) -> np.ndarray:
    """Scale intrinsics for resized image."""
    K = np.asarray(K, dtype=np.float32).copy()
    sx = float(new_width) / float(orig_width)
    sy = float(new_height) / float(orig_height)
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return K
