"""
Pure JAX differentiable 3D Gaussian splatting renderer for MBD3D.

Implements soft splatting with 2D Gaussian projection for gradient flow.
Used during MBD optimization to match gsplat evaluation.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False

from ..types import SceneParams
from ..protocols import BaseObservationRenderer


def _pose_to_w2c_jax(pose: Any) -> Any:
    """Convert 7D pose [px,py,pz,qw,qx,qy,qz] to 4x4 world-to-camera matrix (JAX)."""
    pose = jnp.asarray(pose)
    t = pose[:3]
    q = pose[3:7]
    R = _quat_wxyz_to_rot_jax(q)
    c2w = jnp.eye(4)
    c2w = c2w.at[:3, :3].set(R)
    c2w = c2w.at[:3, 3].set(t)
    return jnp.linalg.inv(c2w)


def _quat_wxyz_to_rot_jax(q: Any) -> Any:
    """Quaternion (w,x,y,z) to 3x3 rotation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _project_means(means: Any, w2c: Any, K: Any) -> tuple[Any, Any]:
    """Project 3D means to 2D. Returns (xy, z_depth)."""
    ones = jnp.ones((means.shape[0], 1))
    pts = jnp.concatenate([means, ones], axis=-1)
    cam = (w2c[:3, :] @ pts.T).T
    z = cam[:, 2:3] + 1e-6
    xy = cam[:, :2] / z
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * xy[:, 0:1] + cx
    v = fy * xy[:, 1:2] + cy
    return jnp.concatenate([u, v], axis=-1), z


def _cov3d_to_cov2d(means: Any, scales: Any, quats: Any, w2c: Any, K: Any) -> Any:
    """Approximate 2D covariance from 3D Gaussian (scale * rotation)."""
    n = means.shape[0]
    R = _quats_to_rotations(quats)
    S = jnp.exp(jnp.clip(scales, -10.0, 10.0))
    cov3d = jnp.einsum("nij,nj,nkj->nik", R, S ** 2, R)

    cam = (w2c[:3, :3] @ means.T).T
    x, y, z = cam[:, 0], cam[:, 1], cam[:, 2] + 1e-6
    fx, fy = K[0, 0], K[1, 1]

    J = jnp.zeros((n, 2, 3))
    J = J.at[:, 0, 0].set(fx / z)
    J = J.at[:, 0, 2].set(-fx * x / (z * z))
    J = J.at[:, 1, 1].set(fy / z)
    J = J.at[:, 1, 2].set(-fy * y / (z * z))

    cov2d = jnp.einsum("nij,njk,nlk->nil", J, cov3d, J)
    eps = 1e-4
    cov2d = cov2d + eps * jnp.eye(2)[None, :, :]
    return cov2d


def _quats_to_rotations(quats: Any) -> Any:
    """(N,4) wxyz quats to (N,3,3) rotation matrices."""
    q = quats / (jnp.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return jnp.stack([
        jnp.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], axis=-1),
        jnp.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], axis=-1),
        jnp.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], axis=-1),
    ], axis=-2)


def _soft_splat_one_view(
    means: Any, scales: Any, quats: Any, opacities: Any, colors: Any,
    w2c: Any, K: Any, H: int, W: int,
) -> Any:
    """Render one view with soft 2D Gaussian splatting + alpha compositing."""
    means_2d, z_depth = _project_means(means, w2c, K)
    cov2d = _cov3d_to_cov2d(means, scales, quats, w2c, K)

    opa = jax.nn.sigmoid(jnp.clip(opacities.ravel(), -50, 50))
    col = jnp.clip(colors, 0.0, 1.0)

    inv_cov = jnp.linalg.inv(cov2d)
    z = jnp.squeeze(z_depth)

    y_coords = jnp.arange(H, dtype=jnp.float32)
    x_coords = jnp.arange(W, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(y_coords, x_coords, indexing="ij")
    pixels = jnp.stack([xx, yy], axis=-1)

    d = pixels[:, :, None, :] - means_2d[None, None, :, :]
    quad = jnp.einsum("hwnd,ndf,hwnf->hwn", d, inv_cov, d)
    alpha = opa[None, None, :] * jnp.exp(-0.5 * jnp.clip(quad, 0.0, 50.0))

    # Cull splats behind camera and clamp alpha for numerical stability.
    alpha = alpha * (z[None, None, :] > 1e-4).astype(alpha.dtype)
    alpha = jnp.clip(alpha, 0.0, 0.995)

    # Approximate gsplat front-to-back compositing using global depth order.
    order = jax.lax.stop_gradient(jnp.argsort(z))
    alpha_s = alpha[:, :, order]
    col_s = col[order]

    one_minus_alpha = 1.0 - alpha_s
    trans = jnp.cumprod(
        jnp.concatenate(
            [
                jnp.ones((*alpha_s.shape[:2], 1), dtype=alpha_s.dtype),
                one_minus_alpha + 1e-8,
            ],
            axis=-1,
        ),
        axis=-1,
    )[:, :, :-1]
    weights = trans * alpha_s
    rgb = jnp.sum(weights[:, :, :, None] * col_s[None, None, :, :], axis=2)
    return rgb


def _soft_splat_multi_view(
    means: Any, scales: Any, quats: Any, opacities: Any, colors: Any,
    w2cs: Any, Ks: Any, H: int, W: int,
) -> Any:
    """Render multiple views with vmap over per-view splatting."""
    render_one = lambda w2c, K: _soft_splat_one_view(
        means, scales, quats, opacities, colors, w2c, K, H, W
    )
    return jax.vmap(render_one, in_axes=(0, 0), out_axes=0)(w2cs, Ks)


class JaxSplatRenderer(BaseObservationRenderer):
    """
    Pure JAX differentiable 3DGS renderer.

    Uses soft 2D Gaussian splatting for gradient flow during MBD optimization.
    Renders match gsplat format for evaluation.
    """

    def __init__(
        self,
        image_height: int = 128,
        image_width: int = 128,
        image_channels: int = 3,
    ):
        if not JAX_AVAILABLE:
            raise ImportError("JAX required for JaxSplatRenderer")
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        self._render_fn = jax.jit(_soft_splat_one_view, static_argnums=(7, 8))
        self._render_multi_fn = jax.jit(_soft_splat_multi_view, static_argnums=(7, 8))
        self._pose_to_w2c_batched = jax.jit(jax.vmap(_pose_to_w2c_jax, in_axes=0, out_axes=0))

    def camera_poses_to_w2c(self, camera_poses: Any) -> Any:
        """Convert a batch of camera poses to world-to-camera matrices."""
        poses = jnp.asarray(camera_poses, dtype=jnp.float32)
        if poses.ndim == 1:
            poses = jnp.reshape(poses, (1, -1))
        return self._pose_to_w2c_batched(poses)

    def render(
        self,
        params: SceneParams,
        camera_poses: Any,
        intrinsics: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Render scene at camera poses. Returns (N, H, W, C) JAX array.
        """
        use_jax = hasattr(params.means, "block_until_ready") or hasattr(camera_poses, "block_until_ready")
        if not use_jax:
            params = _to_jax_params(params)
            camera_poses = jnp.asarray(camera_poses)

        n_views = camera_poses.shape[0] if hasattr(camera_poses, "shape") else len(camera_poses)
        if hasattr(camera_poses, "shape") and camera_poses.ndim == 1:
            n_views = 1
            camera_poses = jnp.reshape(camera_poses, (1, -1))

        means = jnp.asarray(params.means)
        scales = jnp.asarray(params.scales)
        quats = jnp.asarray(params.quats)
        opacities = jnp.asarray(params.opacities)
        colors = jnp.asarray(params.colors)
        n = means.shape[0]
        if opacities.size != n:
            opacities = jnp.broadcast_to(opacities, (n,)).reshape(n, -1)
        if colors.shape[0] != n:
            colors = jnp.broadcast_to(colors, (n, 3))

        if intrinsics is not None:
            K = jnp.asarray(intrinsics)
            if K.ndim == 2:
                K = jnp.tile(K[None, :, :], (n_views, 1, 1))
        else:
            render_downsample = int(max(1, kwargs.get("render_downsample", 1)))
            target_w = max(1, self.image_width // render_downsample)
            target_h = max(1, self.image_height // render_downsample)
            f = 0.5 * target_w / jnp.tan(0.5 * 0.691)
            K = jnp.array([[[f, 0, target_w / 2],
                           [0, f, target_h / 2],
                           [0, 0, 1]]], dtype=jnp.float32)
            K = jnp.tile(K, (n_views, 1, 1))

        camera_w2c = kwargs.get("camera_w2c", None)
        if camera_w2c is None:
            w2cs = self._pose_to_w2c_batched(camera_poses)
        else:
            w2cs = jnp.asarray(camera_w2c, dtype=jnp.float32)
            if w2cs.ndim == 2:
                w2cs = w2cs[None, :, :]
        render_downsample = int(max(1, kwargs.get("render_downsample", 1)))
        target_h = max(1, self.image_height // render_downsample)
        target_w = max(1, self.image_width // render_downsample)
        out = self._render_multi_fn(
            means, scales, quats, opacities, colors,
            w2cs, K, target_h, target_w,
        )
        if not use_jax:
            return np.asarray(out)
        return out

    def supports_jax(self) -> bool:
        return JAX_AVAILABLE


def _to_jax_params(params: SceneParams) -> SceneParams:
    """Convert numpy SceneParams to JAX arrays."""
    return SceneParams(
        means=jnp.asarray(params.means),
        scales=jnp.asarray(params.scales),
        quats=jnp.asarray(params.quats),
        opacities=jnp.asarray(params.opacities),
        colors=jnp.asarray(params.colors),
        spherical_harmonics=jnp.asarray(params.spherical_harmonics) if params.spherical_harmonics is not None else None,
    )
