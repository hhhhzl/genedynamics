# Derived from Yifan Hou ACP (MIT), commit 74d68eca3bfb5cb59da131c6351d9e1c9b129ff6.
# Copyright (c) 2024 Yifan Hou@Stanford. See LICENSE and SOURCE.json.
"""ACP numpy coordinates: SI translation, row-based rotation6d, (linear, angular) twists."""
import numpy as np

def transpose(mat):
    """
    Transpose the last two dimensions of a batch of matrices.

    :param      mat:  (..., h, w) np array
    :return:     (...,  w, h) np array, transposed matrices
    """
    return np.swapaxes(mat, -1, -2)

def normalize(vec, eps=1e-12):
    """
    Normalize vectors along the last dimension.
    args:
        vec: (..., N) np array
    return:
        (..., N) np array of the same shape
    """
    norm = np.linalg.norm(vec, axis=-1)  # (...)
    norm = np.maximum(norm, eps)
    out = vec / norm[..., np.newaxis]
    return out

def wedge3(vec):
    """
    Compute the skew-symmetric wedge matrix of a batch of 3D vectors.

    :param      vec:  (..., 3) np array
    :return:     (..., 3, 3) np array, skew-symmetric matrix
    """
    shape = vec.shape[:-1]
    out = np.zeros(shape + (3, 3), dtype=vec.dtype)
    out[..., 0, 1] = -vec[..., 2]
    out[..., 0, 2] = vec[..., 1]
    out[..., 1, 2] = -vec[..., 0]

    out[..., 1, 0] = -out[..., 0, 1]
    out[..., 2, 0] = -out[..., 0, 2]
    out[..., 2, 1] = -out[..., 1, 2]
    return out

def SE3_inv(mat):
    """
    Efficient inverse of a batch of SE3 matrices.

    Tested by generating random SE3 and verify SE3_inv(SE3) @ SE3 = Identity.

    :param      mat:  (..., 4, 4) np array
    :return:     (..., 4, 4) np array, inverse of the input matrix
    """
    SE3_inv = np.zeros_like(mat)
    SE3_inv[..., :3, :3] = transpose(mat[..., :3, :3])

    temp = -SE3_inv[..., :3, :3] @ np.expand_dims(mat[..., :3, 3], -1)
    # Preserve singleton batch/arm/horizon axes; the native squeeze() removes them.
    SE3_inv[..., :3, 3] = temp.squeeze(-1)
    SE3_inv[..., 3, 3] = 1
    return SE3_inv

def pose9_to_SE3(d9):
    p = d9[..., :3]
    d6 = d9[..., 3:]
    R = rot6_to_SO3(d6)
    SE3 = np.zeros(d9.shape[:-1] + (4, 4), dtype=d9.dtype)
    SE3[..., :3, :3] = R
    SE3[..., :3, 3] = p
    SE3[..., 3, 3] = 1
    return SE3

def rot6_to_SO3(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    SO3 = np.stack((b1, b2, b3), axis=-2)
    return SO3

def SE3_to_adj(SE3):
    """
    Compute the adjoint matrix of a batch of SE3 matrices.

    :param      SE3:  (..., 4, 4) np array
    :return:     (..., 6, 6) np array, adjoint matrices
    """
    shape = SE3.shape[:-2]
    R = SE3[..., :3, :3]
    p = SE3[..., :3, 3]
    Adj = np.zeros(shape + (6, 6), dtype=SE3.dtype)
    Adj[..., :3, :3] = R
    Adj[..., :3, 3:] = wedge3(p) @ R
    Adj[..., 3:, 3:] = R
    return Adj

def SE3_to_pose9(SE3):
    p = SE3[..., :3, 3]
    R = SE3[..., :3, :3]
    d6 = SO3_to_rot6d(R)
    pose9 = np.concatenate([p, d6], axis=-1)
    return pose9

def SO3_to_rot6d(mat):
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out

def so3_to_SO3(v, kEpsilon=1e-9):
    """Get rotation matrix from exponential coordinate

    :param      v:    (..., 3) numpy array

    :returns:   (..., 3, 3) numpy array

    Needs to be tested
    """
    theta = np.linalg.norm(v, axis=-1, keepdims=True)
    theta = np.maximum(theta, kEpsilon)
    vn = v / theta
    v_wedge = wedge3(vn)
    SO3 = (
        np.eye(3) + v_wedge * np.sin(theta) + v_wedge @ v_wedge * (1.0 - np.cos(theta))
    )
    return SO3

def spt_to_SE3(twist):
    """special twist coordinate to displacement
    :param      twist:  (..., 6) np array, twist coordinates
    :return      (..., 4, 4) np array
    """
    if twist.shape == (6, 1):
        twist = twist.reshape(6)  # (6,1) -> (6,)
    if twist.shape[-1] != 6:
        raise ValueError(f"twist shape should be (..., 6) or (6,), got {twist.shape}")
    SE3 = np.zeros(twist.shape[:-1] + (4, 4), dtype=twist.dtype)
    SE3[..., :3, :3] = so3_to_SO3(twist[..., 3:])
    SE3[..., :3, 3] = twist[..., :3]
    SE3[..., 3, 3] = 1
    return SE3
