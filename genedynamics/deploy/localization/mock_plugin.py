"""
Mock localization plugin for testing and sim2real without hardware.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from genedynamics.deploy.localization.base_plugin import BaseLocalizationPlugin


class MockLocalizationPlugin(BaseLocalizationPlugin):
    """Mock plugin: returns fixed or configurable pose/velocity."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        # Default: standing quadruped pose
        self._qpos = np.array(config.get("qpos", [
            0.0, 0.0, 0.5,  # x, y, z
            1.0, 0.0, 0.0, 0.0,  # qw, qx, qy, qz
        ]), dtype=np.float64)
        if self._qpos.size == 7:
            pass
        elif self._qpos.size >= 13:
            self._qpos = self._qpos[:7]
        else:
            self._qpos = np.pad(self._qpos, (0, max(0, 7 - self._qpos.size)), mode="edge")[:7]
        self._qvel = np.array(config.get("qvel", [0.0] * 6), dtype=np.float64)
        self._qvel = np.pad(self._qvel, (0, max(0, 6 - self._qvel.size)), mode="edge")[:6]
        self._last_time = time.time()

    def get_state(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        self._last_time = time.time()
        return self._qpos.copy(), self._qvel.copy()

    def get_last_update_time(self) -> Optional[float]:
        return self._last_time

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """Update mock state (for testing)."""
        self._qpos = np.asarray(qpos, dtype=np.float64).ravel()[:7]
        self._qvel = np.asarray(qvel, dtype=np.float64).ravel()[:6]
        self._last_time = time.time()
