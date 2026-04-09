"""Joint position limit filter.

Clips :class:`ControlCommand.joint_pos` (and the ``joint_pos`` slice of a
``"mixed"`` command) to the per-joint range declared in the
:class:`G1RobotSpec` (or any compatible spec exposing ``joint_range``).

Joints absent from the command are left untouched. The filter is stateless
and runs in O(num_actuated). It records the largest violation (in radians)
per filter pass under :attr:`SafetyResult.violations`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from genedynamics.deploy.interfaces.messages import ControlCommand
from genedynamics.deploy.interfaces.safety import SafetyResult
from genedynamics.deploy.safety.base import BaseSafetyFilter

__all__ = ["JointLimitFilter"]


class JointLimitFilter(BaseSafetyFilter):
    """Clip joint position commands to the spec's joint ranges.

    Args:
        spec: A :class:`G1RobotSpec` (or compatible) exposing
            ``actuated_joints`` and ``joint_range``.
        margin: Optional safety margin in radians shrinking the active
            range on both sides. Use this to keep the controller away from
            hardware-side soft stops.
    """

    def __init__(self, spec: Any, *, margin: float = 0.0) -> None:
        super().__init__(spec)
        self.margin = float(margin)
        n = len(spec.actuated_joints)
        lo = np.full(n, -np.inf, dtype=np.float64)
        hi = np.full(n, +np.inf, dtype=np.float64)
        for i, name in enumerate(spec.actuated_joints):
            jr = spec.joint_range.get(name)
            if jr is None:
                continue
            lo[i] = float(jr[0]) + self.margin
            hi[i] = float(jr[1]) - self.margin
        self._lo = lo
        self._hi = hi

    def filter(self, state, cmd: ControlCommand) -> SafetyResult:  # noqa: D401
        if cmd.kind not in ("joint_pos", "mixed") or cmd.joint_pos is None:
            return self._passthrough(cmd)
        jp = np.asarray(cmd.joint_pos, dtype=np.float64).reshape(-1)
        n = min(jp.size, self._lo.size)
        jp_clipped = jp.copy()
        jp_clipped[:n] = np.clip(jp[:n], self._lo[:n], self._hi[:n])
        diff = jp[:n] - jp_clipped[:n]
        max_violation = float(np.max(np.abs(diff))) if n > 0 else 0.0
        intervened = max_violation > 0.0
        new_cmd = ControlCommand(
            kind=cmd.kind,
            joint_pos=jp_clipped,
            joint_vel=cmd.joint_vel,
            joint_torque=cmd.joint_torque,
            kp=cmd.kp,
            kd=cmd.kd,
            loco_cmd=cmd.loco_cmd,
            extras=cmd.extras,
        )
        return SafetyResult(
            command=new_cmd,
            intervened=intervened,
            violations={"joint_limit_max_rad": max_violation} if intervened else {},
        )
