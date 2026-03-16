"""
Adapter: use localization plugin as state source for real deployment.

Fuses base pose/velocity from localization with joint state from robot (or default).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from genedynamics.deploy.localization.base_plugin import BaseLocalizationPlugin


class LocalizationStateBackend:
    """
    State backend that uses localization plugin for base pose/velocity.

    For full state: base (7+6) from plugin, joints from get_joint_state callback
    or default (zeros). Used when robot SDK doesn't provide base pose (e.g. odom).
    """

    def __init__(
        self,
        plugin: BaseLocalizationPlugin,
        nq: int = 19,
        nv: int = 18,
        get_joint_state: Optional[Callable[[], tuple]] = None,
    ) -> None:
        self.plugin = plugin
        self.nq = nq
        self.nv = nv
        self._get_joint_state = get_joint_state
        self._last_time: Optional[float] = None

    def get_qpos_qvel(self) -> tuple:
        base = self.plugin.get_state()
        if base is None:
            raise RuntimeError("No state from localization plugin")
        qpos_base, qvel_base = base
        qpos_base = np.asarray(qpos_base, dtype=np.float64).ravel()
        qvel_base = np.asarray(qvel_base, dtype=np.float64).ravel()
        if qpos_base.size >= 7 and qvel_base.size >= 6:
            qpos_b = qpos_base[:7]
            qvel_b = qvel_base[:6]
        else:
            raise ValueError("Localization must return at least 7+6 values")

        if self._get_joint_state:
            qpos_j, qvel_j = self._get_joint_state()
        else:
            n_j = self.nq - 7
            qpos_j = np.zeros(n_j, dtype=np.float64)
            qvel_j = np.zeros(self.nv - 6, dtype=np.float64)

        qpos = np.concatenate([qpos_b, np.asarray(qpos_j, dtype=np.float64).ravel()[: self.nq - 7]])
        qvel = np.concatenate([qvel_b, np.asarray(qvel_j, dtype=np.float64).ravel()[: self.nv - 6]])
        self._last_time = self.plugin.get_last_update_time()
        return qpos, qvel

    def get_timestamp(self) -> float:
        t = self.plugin.get_last_update_time()
        return t if t is not None else 0.0
