"""Zero-physics stub :class:`RobotIO` for tests and dry runs.

:class:`StubRobotIO` accepts every command kind, never moves, and emits a
constant :class:`RobotState`. It exists so the runner / config schema /
preset infrastructure can be tested end-to-end without dragging in MuJoCo
or the Unitree SDK.

The stub records every command it receives in :attr:`sent_commands` so
tests can assert on what the controller produced.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from genedynamics.deploy.interfaces.messages import ControlCommand, RobotState
from genedynamics.deploy.io.base import BaseRobotIO

__all__ = ["StubRobotIO", "StubSpec"]


class StubSpec:
    """Minimal spec stand-in: ``num_actuated`` joints with no metadata."""

    def __init__(self, num_actuated: int = 4) -> None:
        self.actuated_joints = tuple(f"j{i}" for i in range(num_actuated))
        self.joint_range = {
            name: np.array([-1.0, 1.0], dtype=np.float64) for name in self.actuated_joints
        }
        self.torque_limit = {name: 10.0 for name in self.actuated_joints}
        self.num_actuated = int(num_actuated)

    @property
    def actuated_qpos_indices(self) -> np.ndarray:
        return np.arange(7, 7 + self.num_actuated, dtype=np.int32)


class StubRobotIO(BaseRobotIO):
    """No-op :class:`RobotIO` for testing the deploy loop without physics.

    Args:
        num_actuated: Number of fake joints. Defaults to 4.
        spec: Optional spec override. When omitted, a fresh
            :class:`StubSpec` is built.
        initial_qpos: Optional ``(7+num_actuated,)`` array used for both
            :meth:`reset` and every subsequent :meth:`get_state`. Defaults
            to zeros.
    """

    physics_backend = "stub"
    array_runtime = "numpy"
    accepts: Tuple[str, ...] = ("joint_pos", "joint_vel", "torque", "loco", "mixed")

    def __init__(
        self,
        *,
        num_actuated: int = 4,
        spec: Optional[StubSpec] = None,
        initial_qpos: Optional[np.ndarray] = None,
    ) -> None:
        spec = spec or StubSpec(num_actuated)
        super().__init__(spec=spec)
        n = spec.num_actuated
        if initial_qpos is None:
            self._qpos = np.zeros(7 + n, dtype=np.float64)
        else:
            self._qpos = np.asarray(initial_qpos, dtype=np.float64).reshape(-1).copy()
        self._qvel = np.zeros(6 + n, dtype=np.float64)
        self.sent_commands: List[ControlCommand] = []
        self.reset_count: int = 0
        self.step_called_count: int = 0

    def _reset_robot(self) -> None:
        self.reset_count += 1
        self.sent_commands.clear()

    def _read_state(self, t: float) -> RobotState:
        return RobotState(
            t=float(t),
            qpos=self._qpos.copy(),
            qvel=self._qvel.copy(),
            base_pose=self._qpos[:7].copy() if self._qpos.size >= 7 else None,
            base_twist=self._qvel[:6].copy() if self._qvel.size >= 6 else None,
            extras={"step": self._step_count},
        )

    def _apply_command(self, cmd: ControlCommand) -> None:
        self.sent_commands.append(cmd)

    def _step_physics(self, dt: float) -> None:
        self.step_called_count += 1

    def _on_close(self) -> None:
        pass
