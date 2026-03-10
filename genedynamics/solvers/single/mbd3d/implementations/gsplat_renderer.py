"""
Gsplat-based 3DGS renderer for MBD3D.

Uses gsplat (PyTorch) for real differentiable 3D Gaussian splatting.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..types import SceneParams
from ..protocols import BaseObservationRenderer

GSPLAT_AVAILABLE = False
try:
    import torch
    import gsplat
    GSPLAT_AVAILABLE = True
except ImportError:
    torch = None
    gsplat = None


def _pose_to_w2c(pose: np.ndarray) -> np.ndarray:
    """Convert 7D pose [px,py,pz,qw,qx,qy,qz] to 4x4 world-to-camera matrix."""
    from ..data.camera_utils import pose_to_c2w_matrix
    c2w = pose_to_c2w_matrix(pose)
    return np.linalg.inv(c2w).astype(np.float32)


class GsplatRenderer(BaseObservationRenderer):
    """
    Real 3DGS renderer via gsplat.

    Converts SceneParams to gsplat format and renders.
    """

    def __init__(
        self,
        image_height: int = 128,
        image_width: int = 128,
        image_channels: int = 3,
    ):
        if not GSPLAT_AVAILABLE:
            raise ImportError("gsplat required. pip install gsplat")
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels

    def _to_numpy(self, x: Any) -> np.ndarray:
        """Convert JAX/DeviceArray to numpy, blocking if needed."""
        if hasattr(x, "block_until_ready"):
            x.block_until_ready()
        return np.asarray(x, dtype=np.float32)

    def render(
        self,
        params: SceneParams,
        camera_poses: Any,
        intrinsics: Optional[Any] = None,
        batch_size: Optional[int] = 32,
        **kwargs: Any,
    ) -> Any:
        """
        Render scene at camera poses using gsplat.

        camera_poses: (N, 7) [px,py,pz,qw,qx,qy,qz]
        intrinsics: (N, 3, 3) or (3, 3)
        batch_size: max views per batch (None = all at once)
        """
        n_views = camera_poses.shape[0] if hasattr(camera_poses, "shape") else len(camera_poses)
        if hasattr(camera_poses, "shape") and camera_poses.ndim == 1:
            n_views = 1
            camera_poses = np.reshape(self._to_numpy(camera_poses), (1, -1))
        else:
            camera_poses = self._to_numpy(camera_poses)

        if batch_size and n_views > batch_size:
            out_list = []
            for i in range(0, n_views, batch_size):
                end = min(i + batch_size, n_views)
                sub_poses = camera_poses[i:end]
                sub_K = None
                if intrinsics is not None:
                    K = self._to_numpy(intrinsics)
                    if K.ndim == 3:
                        sub_K = K[i:end]
                    else:
                        sub_K = K
                sub_out = self._render_batch(params, sub_poses, sub_K)
                out_list.append(sub_out)
            return np.concatenate(out_list, axis=0)

        return self._render_batch(params, camera_poses, intrinsics)

    def _render_batch(
        self,
        params: SceneParams,
        camera_poses: np.ndarray,
        intrinsics: Optional[Any],
    ) -> np.ndarray:
        """Render a single batch of views."""
        n_views = camera_poses.shape[0]
        means = self._to_numpy(params.means).copy()
        scales = self._to_numpy(params.scales).copy()
        quats = self._to_numpy(params.quats).copy()
        opacities = self._to_numpy(params.opacities).ravel().copy()
        colors = self._to_numpy(params.colors).copy()

        n = means.shape[0]
        if opacities.size != n:
            opacities = np.broadcast_to(opacities, (n,)).ravel()
        if colors.shape[0] != n:
            colors = np.broadcast_to(colors, (n, 3))

        scales_pos = np.exp(np.clip(scales, -10.0, 10.0))
        opacities_sigm = 1.0 / (1.0 + np.exp(-np.clip(opacities, -50, 50)))
        colors_clip = np.clip(colors, 0.0, 1.0)

        means_t = torch.from_numpy(means).float()
        quats_t = torch.from_numpy(quats).float()
        scales_t = torch.from_numpy(scales_pos).float()
        opacities_t = torch.from_numpy(opacities_sigm).float()
        colors_t = torch.from_numpy(colors_clip).float()

        viewmats = np.stack([_pose_to_w2c(camera_poses[i]) for i in range(n_views)], axis=0)
        if intrinsics is not None:
            K = self._to_numpy(intrinsics)
            if K.ndim == 2:
                K = np.tile(K[None, :, :], (n_views, 1, 1))
        else:
            f = 0.5 * self.image_width / np.tan(0.5 * 0.691)
            K = np.array([[[f, 0, self.image_width / 2],
                           [0, f, self.image_height / 2],
                           [0, 0, 1]]], dtype=np.float32)
            K = np.tile(K, (n_views, 1, 1))

        viewmats_t = torch.from_numpy(viewmats).float()
        Ks_t = torch.from_numpy(K).float()
        if torch.cuda.is_available():
            means_t = means_t.cuda()
            quats_t = quats_t.cuda()
            scales_t = scales_t.cuda()
            opacities_t = opacities_t.cuda()
            colors_t = colors_t.cuda()
            viewmats_t = viewmats_t.cuda()
            Ks_t = Ks_t.cuda()

        rgb, _, _ = gsplat.rasterization(
            means_t, quats_t, scales_t, opacities_t, colors_t,
            viewmats_t, Ks_t,
            self.image_width, self.image_height,
        )
        out = rgb.detach().cpu().numpy()
        if out.ndim == 4 and out.shape[1] == 3:
            out = np.transpose(out, (0, 2, 3, 1))
        if out.ndim == 4 and out.shape[0] == n_views:
            return out
        if out.ndim == 3:
            return out[None, ...]
        return out

    def supports_jax(self) -> bool:
        return False
