"""Torque magnitude filter.

Clips :class:`ControlCommand.joint_torque` to the per-joint magnitude bound
declared in the spec's ``actuator_forcerange``. The filter handles every
command kind that carries a feedforward torque (``"torque"``,
``"joint_pos"`` with ff term, ``"mixed"``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from genedynamics.deploy.interfaces.messages import ControlCommand
from genedynamics.deploy.interfaces.safety import SafetyResult
from genedynamics.deploy.safety.base import BaseSafetyFilter

__all__ = ["TorqueLimitFilter"]


class TorqueLimitFilter(BaseSafetyFilter):
    """Clip torque commands to the spec's actuator force bounds.

    Args:
        spec: Robot spec exposing ``actuated_joints`` and ``torque_limit``.
        safety_margin: Multiplier in (0, 1] applied to the spec limits.
            ``0.9`` keeps a 10% headroom relative to the MJCF declaration.
    """

    def __init__(self, spec: Any, *, safety_margin: float = 1.0) -> None:
        super().__init__(spec)
        if not (0.0 < safety_margin <= 1.0):
            raise ValueError(
                f"torque safety_margin must be in (0, 1], got {safety_margin}"
            )
        self.safety_margin = float(safety_margin)
        n = len(spec.actuated_joints)
        bounds = np.full(n, np.inf, dtype=np.float64)
        for i, name in enumerate(spec.actuated_joints):
            tl = spec.torque_limit.get(name, np.inf)
            bounds[i] = float(tl) * self.safety_margin
        self._bounds = bounds

    def filter(self, state, cmd: ControlCommand) -> SafetyResult:
        if cmd.joint_torque is None:
            return self._passthrough(cmd)
        tau = np.asarray(cmd.joint_torque, dtype=np.float64).reshape(-1)
        n = min(tau.size, self._bounds.size)
        tau_clipped = tau.copy()
        tau_clipped[:n] = np.clip(tau[:n], -self._bounds[:n], self._bounds[:n])
        diff = tau[:n] - tau_clipped[:n]
        max_violation = float(np.max(np.abs(diff))) if n > 0 else 0.0
        intervened = max_violation > 0.0
        new_cmd = ControlCommand(
            kind=cmd.kind,
            joint_pos=cmd.joint_pos,
            joint_vel=cmd.joint_vel,
            joint_torque=tau_clipped,
            kp=cmd.kp,
            kd=cmd.kd,
            loco_cmd=cmd.loco_cmd,
            extras=cmd.extras,
        )
        return SafetyResult(
            command=new_cmd,
            intervened=intervened,
            violations={"torque_max_nm": max_violation} if intervened else {},
        )
