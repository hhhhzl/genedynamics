"""
Mock observation renderer for MBD3D testing and development.

Replace with differentiable 3DGS renderer (e.g., diff-gaussian-splatting)
for production use.
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


class MockRenderer(BaseObservationRenderer):
    """
    Placeholder renderer that returns differentiable surrogate.

    Instead of actual splatting, returns a simple function of scene params
    and camera poses for gradient flow. Use for:
    - Unit tests
    - Pipeline development
    - Ablation (prior-only vs full likelihood)

    Production: swap with DiffSplatRenderer or similar.
    """

    def __init__(
        self,
        image_height: int = 64,
        image_width: int = 64,
        image_channels: int = 3,
        surrogate_type: str = "linear",
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        self.surrogate_type = surrogate_type
        self._pixels_per_view = image_height * image_width * image_channels

    def render(
        self,
        params: SceneParams,
        camera_poses: Any,
        intrinsics: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Return surrogate "rendered" images.

        Surrogate: simple differentiable function so gradients exist.
        - linear: mean(means) broadcast + 0.1 * camera_poses (flattened)
        - quadratic: adds squared terms for curvature
        """
        use_jax = JAX_AVAILABLE and (
            hasattr(params.means, "block_until_ready") or hasattr(camera_poses, "block_until_ready")
        )
        n_views = camera_poses.shape[0] if hasattr(camera_poses, "shape") else len(camera_poses)
        if hasattr(camera_poses, "shape") and camera_poses.ndim == 1:
            n_views = 1
            camera_poses = camera_poses[None, :] if use_jax else np.reshape(camera_poses, (1, -1))

        if use_jax:
            cam_flat = jnp.reshape(jnp.asarray(camera_poses), (n_views, -1))
            mean_scene = jnp.mean(params.means)
            # (N, H*W*C) surrogate
            base = jnp.broadcast_to(mean_scene, (n_views, self._pixels_per_view))
            pose_contrib = 0.1 * jnp.tile(cam_flat[:, : min(7, cam_flat.shape[1])], (1, (self._pixels_per_view // 7) + 1))[
                :, : self._pixels_per_view
            ]
            out = base + pose_contrib
            if self.surrogate_type == "quadratic":
                out = out + 0.01 * jnp.square(out)
            return jnp.reshape(out, (n_views, self.image_height, self.image_width, self.image_channels))
        else:
            cam_flat = np.reshape(np.asarray(camera_poses), (n_views, -1))
            mean_scene = float(np.mean(params.means))
            base = np.broadcast_to(mean_scene, (n_views, self._pixels_per_view)).copy()
            pose_contrib = np.tile(
                cam_flat[:, : min(7, cam_flat.shape[1])],
                (1, (self._pixels_per_view // 7) + 1),
            )[:, : self._pixels_per_view] * 0.1
            out = base + pose_contrib
            if self.surrogate_type == "quadratic":
                out = out + 0.01 * np.square(out)
            return np.reshape(out, (n_views, self.image_height, self.image_width, self.image_channels))

    def supports_jax(self) -> bool:
        return JAX_AVAILABLE
