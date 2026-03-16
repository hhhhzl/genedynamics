"""
JAX-native renderer backed by external jaxsplat kernels.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax.lib import xla_client as _xla_client
    from jaxlib import xla_client as _jaxlib_xla
    from jax.extend import core as _ext_core

    # Compatibility patch for jaxsplat with JAX >= 0.6.
    if not hasattr(jax.core, "canonicalize_shape"):
        from jax._src.core import canonicalize_shape as _canonicalize_shape
        setattr(jax.core, "canonicalize_shape", _canonicalize_shape)
    if not hasattr(_xla_client, "register_custom_call_target") and hasattr(
        _jaxlib_xla, "register_custom_call_target"
    ):
        setattr(
            _xla_client,
            "register_custom_call_target",
            _jaxlib_xla.register_custom_call_target,
        )
    if not hasattr(jax.core, "Primitive"):
        setattr(jax.core, "Primitive", _ext_core.Primitive)

    import jaxsplat

    JAXSPLAT_AVAILABLE = True
except Exception:
    jax = None
    jnp = None
    jaxsplat = None
    JAXSPLAT_AVAILABLE = False

from ..types import SceneParams
from ..protocols import BaseObservationRenderer


def _pose_to_w2c_jax(pose: Any) -> Any:
    pose = jnp.asarray(pose, dtype=jnp.float32)
    t = pose[:3]
    q = pose[3:7]
    q = q / (jnp.linalg.norm(q) + 1e-8)
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = jnp.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=jnp.float32,
    )
    c2w = jnp.eye(4, dtype=jnp.float32)
    c2w = c2w.at[:3, :3].set(R)
    c2w = c2w.at[:3, 3].set(t)
    return jnp.linalg.inv(c2w)


class JaxsplatRenderer(BaseObservationRenderer):
    """Renderer wrapper using external jaxsplat CUDA kernels."""

    def __init__(
        self,
        image_height: int = 128,
        image_width: int = 128,
        image_channels: int = 3,
    ):
        if not JAXSPLAT_AVAILABLE:
            raise ImportError("jaxsplat unavailable. Install from GitHub source.")
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        # jaxsplat custom primitives currently do not implement vmap batching rules.
        self.requires_non_batched_logpi = True
        # jaxsplat custom_vjp currently does not support forward-mode jvp.
        self.requires_no_jvp = True
        self._pose_to_w2c_batched = jax.jit(jax.vmap(_pose_to_w2c_jax, in_axes=0, out_axes=0))

    def render(
        self,
        params: SceneParams,
        camera_poses: Any,
        intrinsics: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        use_jax = hasattr(params.means, "block_until_ready") or hasattr(camera_poses, "block_until_ready")
        if not use_jax:
            camera_poses = jnp.asarray(camera_poses, dtype=jnp.float32)
            means = jnp.asarray(params.means, dtype=jnp.float32)
            scales_raw = jnp.asarray(params.scales, dtype=jnp.float32)
            quats = jnp.asarray(params.quats, dtype=jnp.float32)
            opacities_raw = jnp.asarray(params.opacities, dtype=jnp.float32)
            colors_raw = jnp.asarray(params.colors, dtype=jnp.float32)
        else:
            means = jnp.asarray(params.means)
            scales_raw = jnp.asarray(params.scales)
            quats = jnp.asarray(params.quats)
            opacities_raw = jnp.asarray(params.opacities)
            colors_raw = jnp.asarray(params.colors)

        if camera_poses.ndim == 1:
            camera_poses = jnp.reshape(camera_poses, (1, -1))
        n_views = int(camera_poses.shape[0])

        n = int(means.shape[0])
        opacities_raw = opacities_raw.reshape(-1, 1)
        if opacities_raw.shape[0] != n:
            opacities_raw = jnp.broadcast_to(opacities_raw, (n, 1))
        if colors_raw.shape[0] != n:
            colors_raw = jnp.broadcast_to(colors_raw, (n, 3))

        # SceneParams stores raw/log-space values; convert to renderer domain.
        scales = jnp.exp(jnp.clip(scales_raw, -10.0, 10.0))
        quats = quats / (jnp.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8)
        colors = jnp.clip(colors_raw, 0.0, 1.0)
        opacities = jax.nn.sigmoid(jnp.clip(opacities_raw, -50.0, 50.0))

        render_downsample = int(max(1, kwargs.get("render_downsample", 1)))
        target_h = max(1, self.image_height // render_downsample)
        target_w = max(1, self.image_width // render_downsample)

        # jaxsplat descriptors require host-concrete intrinsics values.
        # Use static focal/center from render resolution to stay JIT-safe.
        f_const = float(0.5 * target_w / np.tan(0.5 * 0.691))
        c_const = (float(target_w) / 2.0, float(target_h) / 2.0)

        camera_w2c = kwargs.get("camera_w2c", None)
        if camera_w2c is None:
            w2cs = self._pose_to_w2c_batched(camera_poses)
        else:
            w2cs = jnp.asarray(camera_w2c, dtype=jnp.float32)
            if w2cs.ndim == 2:
                w2cs = w2cs[None, :, :]

        background = jnp.ones((3,), dtype=jnp.float32)
        images = []
        for i in range(n_views):
            img = jaxsplat.render(
                means,
                scales,
                quats,
                colors,
                opacities,
                viewmat=w2cs[i],
                background=background,
                img_shape=(target_h, target_w),
                f=(f_const, f_const),
                c=c_const,
                glob_scale=1.0,
                clip_thresh=0.01,
                block_size=16,
            )
            images.append(img)
        out = jnp.stack(images, axis=0)
        if use_jax:
            return out
        return np.asarray(out)

    def supports_jax(self) -> bool:
        return JAXSPLAT_AVAILABLE

