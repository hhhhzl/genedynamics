"""
MuJoCo active-perception environment plugin (exp3).

Provides:
  - `ObservationBundle` rendered from a MuJoCo scene XML at candidate camera poses.
  - A reset()/step() interface compatible with the 3DGS runner.

This environment is used by exp3's closed-loop demo, where a planner picks which
candidate camera to move to next and the env renders the corresponding RGB frame.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from genedynamics.data import (
    MuJoCoSceneDataAdapter,
    mujoco_scene_config_from_env_params,
)

from ...framework.base import EnvironmentPlugin
from ._scene_env_base import SceneEnvBase


class MuJoCoActivePerceptionPlugin(EnvironmentPlugin):
    """Environment plugin for exp3 active perception on a MuJoCo scene."""

    @property
    def name(self) -> str:
        return "3dgs_mujoco_active"

    def create_env(self, config: Dict[str, Any]) -> Any:
        return _MuJoCoActivePerceptionEnv(config)

    def create_energy(self, env: Any = None) -> Any:
        from genedynamics.core.energy import LegacyEnergyFunctional, EnergyTerm

        def zero_energy(x, u, ctx):
            return 0.0

        return LegacyEnergyFunctional({"task": EnergyTerm(zero_energy, 0.0)})

    def get_state_dim(self) -> int:
        return 7

    def extract_position(self, state: Any) -> np.ndarray:
        arr = np.asarray(state, dtype=np.float32) if state is not None else np.zeros(3)
        return arr[:3] if arr.size >= 3 else np.pad(arr, (0, 3 - arr.size))


class _MuJoCoActivePerceptionEnv(SceneEnvBase):
    def _make_adapter(self, config: Dict[str, Any]):
        infer = int(config.get("resolution_infer", config.get("image_height", 128)))
        adapter_config = mujoco_scene_config_from_env_params(
            config,
            adapter_default_split=str(config.get("split", "train")),
            image_height=infer,
            image_width=infer,
        )
        return MuJoCoSceneDataAdapter(adapter_config)
