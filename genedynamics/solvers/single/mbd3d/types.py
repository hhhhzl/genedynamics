"""
Type definitions for MBD3D (3DGS robust mapping) solver.

Defines structured types for scene state, observations, and solver results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jnp = None
    JAX_AVAILABLE = False

Array = Union[np.ndarray, Any]


@dataclass
class SceneParams:
    """
    Structured 3D Gaussian Splatting scene parameters.

    Attributes:
        means: (N, 3) Gaussian centers in world coordinates
        scales: (N, 3) scale factors (log-space for positivity)
        quats: (N, 4) quaternions for rotation (xyzw)
        opacities: (N,) or (N, 1) opacity values (logit-space)
        colors: (N, 3) or (N, C) spherical harmonics / RGB
        spherical_harmonics: optional (N, K) SH coefficients for view-dependent color
    """

    means: Array
    scales: Array
    quats: Array
    opacities: Array
    colors: Array
    spherical_harmonics: Optional[Array] = None

    def flatten(self) -> Array:
        """Flatten to 1D vector for optimization."""
        use_jax = JAX_AVAILABLE and hasattr(self.means, "block_until_ready")

        def _ravel(x: Array) -> Array:
            a = jnp.asarray(x) if use_jax else np.asarray(x)
            return a.ravel()

        parts = [_ravel(self.means), _ravel(self.scales), _ravel(self.quats), _ravel(self.opacities), _ravel(self.colors)]
        if self.spherical_harmonics is not None:
            parts.append(_ravel(self.spherical_harmonics))

        if use_jax:
            return jnp.concatenate([jnp.ravel(p) for p in parts])
        return np.concatenate([np.ravel(np.asarray(p)) for p in parts])

    @classmethod
    def from_flattened(
        cls,
        flat: Array,
        n_gaussians: int,
        use_sh: bool = False,
        sh_dim: int = 0,
    ) -> "SceneParams":
        """Reconstruct from flattened vector."""
        use_jax = JAX_AVAILABLE and (
            hasattr(flat, "block_until_ready") or getattr(flat, "aval", None) is not None
        )
        flat = jnp.asarray(flat) if use_jax else np.asarray(flat)
        n = n_gaussians
        idx = 0

        def take(size: int):
            nonlocal idx
            s = flat[idx : idx + size]
            idx += size
            return s

        means = take(n * 3).reshape(n, 3)
        scales = take(n * 3).reshape(n, 3)
        quats = take(n * 4).reshape(n, 4)
        opacities = take(n).reshape(n, -1)
        colors = take(n * 3).reshape(n, 3)
        sh = take(n * sh_dim).reshape(n, sh_dim) if use_sh and sh_dim > 0 else None
        return cls(means=means, scales=scales, quats=quats, opacities=opacities, colors=colors, spherical_harmonics=sh)


# CameraPose and ObservationBundle now live in genedynamics.data.types
# (shared across all dataset adapters). Re-exported here for backward
# compatibility with existing `from ...mbd3d.types import ObservationBundle`.
from genedynamics.data.types import CameraPose, ObservationBundle  # noqa: F401


@dataclass
class MBD3DResult:
    """
    Result from MBD3D solve.

    Attributes:
        scene_params: optimized scene parameters
        camera_trajectory: (H+1, pose_dim) camera poses over trajectory
        states: list of states (camera poses) for Trajectory compatibility
        actions: list of actions (pose deltas) for Trajectory compatibility
        diagnostics: MCSA/bridge diagnostics per step
        bridge_history: optional β_k, ESS, etc. per step
        total_log_prob: final log π(θ)
        n_steps: number of bridge steps executed
    """

    scene_params: SceneParams
    camera_trajectory: Array
    states: List[Array]
    actions: List[Array]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    bridge_history: Optional[Dict[str, List[float]]] = None
    total_log_prob: float = 0.0
    n_steps: int = 0

    def to_trajectory_info(self) -> Dict[str, Any]:
        """Convert to dict compatible with Trajectory.info and method plugins."""
        return {
            "scene_params": self.scene_params,
            "camera_trajectory": self.camera_trajectory,
            "diagnostics": self.diagnostics,
            "bridge_history": self.bridge_history,
            "total_log_prob": self.total_log_prob,
            "n_steps": self.n_steps,
        }
