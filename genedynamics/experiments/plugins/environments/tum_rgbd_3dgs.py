"""
TUM RGB-D environment plugin for 3DGS experiments (exp2).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from genedynamics.data import TUMDataAdapter, tum_config_from_env_params

from ...framework.base import EnvironmentPlugin
from ._scene_env_base import SceneEnvBase


class TUM_RGBD_3DGSPlugin(EnvironmentPlugin):
    """Environment plugin for 3DGS reconstruction on TUM RGB-D sequences."""

    @property
    def name(self) -> str:
        return "3dgs_tum_rgbd"

    def create_env(self, config: Dict[str, Any]) -> Any:
        return _TUMRGBD3DGSEnv(config)

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


class _TUMRGBD3DGSEnv(SceneEnvBase):
    def _make_adapter(self, config: Dict[str, Any]):
        infer = int(config.get("resolution_infer", config.get("image_height", 128)))
        adapter_config = tum_config_from_env_params(
            config,
            adapter_default_split=str(config.get("split", "train")),
            image_height=infer,
            image_width=infer,
        )
        return TUMDataAdapter(adapter_config)
