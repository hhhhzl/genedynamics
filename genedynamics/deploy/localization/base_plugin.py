"""
Base localization plugin interface.

All plugins return (qpos, qvel) in world frame. qpos: [x,y,z, qw,qx,qy,qz].
qvel: [vx,vy,vz, wx,wy,wz] in world frame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class BaseLocalizationPlugin(ABC):
    """Base class for localization plugins."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def get_state(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Return (qpos, qvel) in world frame.

        qpos: [x, y, z, qw, qx, qy, qz] (7 floats)
        qvel: [vx, vy, vz, wx, wy, wz] (6 floats) in world frame.

        Returns None if no update received.
        """
        ...

    @abstractmethod
    def get_last_update_time(self) -> Optional[float]:
        """Timestamp of last update. None if never updated."""
        ...

    def health(self) -> str:
        """Health status: ok | stale | failed."""
        t = self.get_last_update_time()
        if t is None:
            return "failed"
        import time
        if time.time() - t > self.config.get("timeout_sec", 2.0):
            return "stale"
        return "ok"
