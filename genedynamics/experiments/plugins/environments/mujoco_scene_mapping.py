"""
Mujoco scene mapping environment plugin for MBD3D (3DGS robust mapping).

Provides a scene (from MuJoCo model or synthetic) and observations for
joint scene + camera trajectory estimation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ...framework.base import EnvironmentPlugin
from genedynamics.solvers.single.mbd3d.types import ObservationBundle


class MujocoSceneMappingPlugin(EnvironmentPlugin):
    """
    Environment plugin for 3DGS robust mapping with MuJoCo scenes.

    State: initial camera pose (7D: position + quat) or None.
    Provides get_observations() -> ObservationBundle for MBD3D likelihood.
    """

    @property
    def name(self) -> str:
        return "mujoco_scene_mapping"

    def create_env(self, config: Dict[str, Any]) -> Any:
        return _MujocoSceneMappingEnv(config)

    def create_energy(self, env: Any = None) -> Any:
        from genedynamics.core.energy import LegacyEnergyFunctional, EnergyTerm

        def zero_energy(x, u, ctx):
            return 0.0

        return LegacyEnergyFunctional({"task": EnergyTerm(zero_energy, 0.0)})

    def get_state_dim(self) -> int:
        return 7

    def extract_position(self, state: Any) -> np.ndarray:
        if state is None or (hasattr(state, "size") and state.size == 0):
            return np.zeros(3, dtype=np.float32)
        arr = np.asarray(state, dtype=np.float32)
        return arr[:3] if arr.size >= 3 else np.pad(arr, (0, 3 - arr.size))


class _MujocoSceneMappingEnv:
    """
    Stub environment for scene mapping.

    Provides:
    - reset() -> (initial_pose, info)
    - get_observations() -> ObservationBundle
    - horizon, dt for planner compatibility
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.horizon = config.get("horizon", 10)
        self.dt = config.get("dt", 0.1)
        self.target = np.zeros(3, dtype=np.float32)  # Dummy for experiment framework
        self._n_views = config.get("n_views", self.horizon + 1)
        self._img_h = config.get("image_height", 64)
        self._img_w = config.get("image_width", 64)
        self._img_c = config.get("image_channels", 3)
        self._rng = np.random.default_rng(config.get("seed", 0))
        self._observations: Optional[ObservationBundle] = None

    def reset(self, rng: Any = None) -> tuple:
        if rng is not None and hasattr(rng, "split"):
            import jax
            key = rng
            pos = jax.random.normal(key, (3,)) * 0.5
            key, _ = jax.random.split(key)
            quat = jax.random.normal(key, (4,))
            quat = quat / (np.linalg.norm(np.asarray(quat)) + 1e-8)
            initial_pose = np.concatenate([np.asarray(pos), np.asarray(quat)], axis=0)
        else:
            initial_pose = np.concatenate([
                self._rng.normal(0, 0.5, 3),
                self._rng.normal(0, 1, 4),
            ], axis=0).astype(np.float32)
            initial_pose[3:7] = initial_pose[3:7] / (np.linalg.norm(initial_pose[3:7]) + 1e-8)
        return initial_pose, {}

    def get_observations(self) -> ObservationBundle:
        if self._observations is not None:
            return self._observations
        n = self._n_views
        images = self._rng.standard_normal((n, self._img_h, self._img_w, self._img_c)).astype(np.float32) * 0.1
        camera_poses = self._rng.standard_normal((n, 7)).astype(np.float32) * 0.1
        camera_poses[:, 3:7] = camera_poses[:, 3:7] / (
            np.linalg.norm(camera_poses[:, 3:7], axis=1, keepdims=True) + 1e-8
        )
        self._observations = ObservationBundle(images=images, camera_poses=camera_poses)
        return self._observations

    def step(self, action: Any) -> tuple:
        return self.reset()[0], 0.0, False, {}
