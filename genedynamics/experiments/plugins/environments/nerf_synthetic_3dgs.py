"""
NeRF Synthetic 3DGS environment plugin for Experiment 1.

Loads NeRF Synthetic data and provides ObservationBundle for MBD3D.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ...framework.base import EnvironmentPlugin
from genedynamics.solvers.single.mbd3d.types import ObservationBundle
from genedynamics.data import (
    NerfSyntheticDataAdapter,
    nerf_synthetic_config_from_env_params,
)


class NerfSynthetic3DGSPlugin(EnvironmentPlugin):
    """
    Environment plugin for 3DGS NeRF Synthetic experiments.

    Creates env that provides get_observations() -> ObservationBundle
    from NeRF Synthetic transforms_*.json.
    """

    @property
    def name(self) -> str:
        return "3dgs_nerf_synthetic"

    def create_env(self, config: Dict[str, Any]) -> Any:
        return _NerfSynthetic3DGSEnv(config)

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


class _NerfSynthetic3DGSEnv:
    """
    Environment that loads NeRF Synthetic and provides ObservationBundle.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.horizon = config.get("horizon", 10)
        self.dt = config.get("dt", 0.1)
        self.target = np.zeros(3, dtype=np.float32)

        dataset_root = config.get("dataset_root") or config.get("dataset_path", "")
        use_stub = config.get("use_stub_when_unavailable", False)

        self._adapter = None
        self._observations: Optional[ObservationBundle] = None

        if dataset_root:
            try:
                from pathlib import Path
                root = Path(dataset_root).expanduser()
                if not root.is_absolute():
                    for base in [Path.cwd(), Path(__file__).resolve().parents[4]]:
                        cand = base / root
                        if cand.exists():
                            root = cand
                            break
                infer = int(config.get("resolution_infer", config.get("image_height", 128)))
                adapter_config = nerf_synthetic_config_from_env_params(
                    config,
                    adapter_default_split=str(config.get("split", "train")),
                    image_height=infer,
                    image_width=infer,
                )
                self._adapter = NerfSyntheticDataAdapter(adapter_config)
            except FileNotFoundError as e:
                if not use_stub:
                    raise
                self._adapter = None

        if self._adapter is None and use_stub:
            self._stub_n_views = config.get("max_views", 11)
            self._stub_h = config.get("resolution_infer", config.get("image_height", 128))
            self._stub_w = config.get("resolution_infer", config.get("image_width", 128))

    def reset(self, rng: Any = None) -> tuple:
        obs = self.get_observations()
        n = obs.num_views
        if n > 0:
            initial_pose = np.asarray(obs.camera_poses[0], dtype=np.float32)
        else:
            initial_pose = np.zeros(7, dtype=np.float32)
            initial_pose[3] = 1.0
        return initial_pose, {}

    def get_observations(self) -> ObservationBundle:
        if self._observations is not None:
            return self._observations
        if self._adapter is not None:
            self._observations = self._adapter.load_bundle(
                split=self.config.get("split", "train"),
            )
        else:
            n = getattr(self, "_stub_n_views", 11)
            h = getattr(self, "_stub_h", 128)
            w = getattr(self, "_stub_w", 128)
            rng = np.random.default_rng(self.config.get("shuffle_seed", 0))
            images = rng.standard_normal((n, h, w, 3)).astype(np.float32) * 0.1 + 0.5
            camera_poses = rng.standard_normal((n, 7)).astype(np.float32) * 0.1
            camera_poses[:, 3:7] = camera_poses[:, 3:7] / (
                np.linalg.norm(camera_poses[:, 3:7], axis=1, keepdims=True) + 1e-8
            )
            self._observations = ObservationBundle(
                images=images,
                camera_poses=camera_poses,
                intrinsics=None,
                timestamps=np.arange(n, dtype=np.float32),
            )
        return self._observations

    def step(self, action: Any) -> tuple:
        return self.reset()[0], 0.0, False, {}
