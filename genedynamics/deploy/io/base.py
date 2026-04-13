"""Base class for sim/real-symmetric robot IO adapters.

Concrete IOs (``MujocoRobotIO``, ``MjxRobotIO``, ``UnitreeG1RobotIO`` …)
inherit :class:`BaseRobotIO` to get:

* Shared :class:`ControlCommand` validation against the IO's ``accepts`` set.
* Latched-command storage so :meth:`step` can apply the most recent command
  written by :meth:`send_control` (matching the spark sim/real symmetric pattern).
* Episode bookkeeping (``episode_id``, step counter, last reset time).

Concrete subclasses must implement the four physics primitives:
:meth:`_reset_robot`, :meth:`_read_state`, :meth:`_apply_command`, and
:meth:`_step_physics`. The minimal :class:`RobotIO` protocol from
:mod:`genedynamics.deploy.interfaces.robot_io` is satisfied automatically.

This module purposely does **not** import any heavy dependency; only the
concrete subclasses (:mod:`mujoco_io`, :mod:`mjx_io`, …) take the hit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

from genedynamics.deploy.interfaces.messages import ControlCommand, RobotState

__all__ = ["BaseRobotIO"]


class BaseRobotIO(ABC):
    """Common scaffolding for every concrete RobotIO.

    Subclasses populate the four ``_reset_robot`` / ``_read_state`` /
    ``_apply_command`` / ``_step_physics`` hooks. The protocol-level methods
    (:meth:`reset`, :meth:`get_state`, :meth:`send_control`, :meth:`step`,
    :meth:`close`) are provided here.

    Attributes:
        spec: Robot spec object (e.g. :class:`G1RobotSpec`). Required so
            controllers can query joint indices, limits and torque bounds
            without depending on the IO subclass.
        physics_backend: Tag identifying the simulator (``"mujoco"`` /
            ``"mjx"`` / ``"brax"`` / ``"isaac"``) or ``None`` for real
            hardware. The pipeline reads it for compatibility checks.
        array_runtime: Tag identifying the array library state arrays come
            out as (``"numpy"`` / ``"jax"`` / ``"torch"``). Controllers
            advertise a matching ``runtime`` so the pipeline asserts
            cross-component compatibility at startup.
        accepts: Tuple of supported :attr:`ControlCommand.kind` values. The
            pipeline verifies the active controller's ``produces`` is a
            subset.
    """

    physics_backend: Optional[str] = None
    array_runtime: str = "numpy"
    accepts: Tuple[str, ...] = ()

    def __init__(self, spec: Any) -> None:
        self.spec = spec
        self._latched_command: Optional[ControlCommand] = None
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._t: float = 0.0
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    @property
    def t(self) -> float:
        """Current internal clock in seconds since the last :meth:`reset`."""
        return self._t

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def episode_id(self) -> Optional[str]:
        return self._episode_id

    def reset(self, episode_id: Optional[str] = None) -> RobotState:
        """Reset the robot, clear latched command, and return the first state."""
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")
        self._latched_command = None
        self._episode_id = episode_id
        self._step_count = 0
        self._t = 0.0
        self._reset_robot()
        return self._read_state(t=self._t)

    # ------------------------------------------------------------------
    # Command latching
    # ------------------------------------------------------------------

    def send_control(self, cmd: ControlCommand) -> None:
        """Latch a command. The next :meth:`step` will apply it.

        Subclasses that drive asynchronous protocols (real hardware) may
        override and dispatch immediately while still calling ``super()``.
        """
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")
        if self.accepts and cmd.kind not in self.accepts:
            raise ValueError(
                f"{type(self).__name__} does not accept ControlCommand.kind="
                f"{cmd.kind!r}; supported kinds: {self.accepts}"
            )
        self._latched_command = cmd

    # ------------------------------------------------------------------
    # State + step
    # ------------------------------------------------------------------

    def get_state(self) -> RobotState:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")
        return self._read_state(t=self._t)

    def step(self, dt: float) -> RobotState:
        """Apply the latched command (if any) and advance the simulation by ``dt``."""
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")
        if dt <= 0.0:
            raise ValueError(f"step dt must be positive, got {dt}")
        if self._latched_command is not None:
            self._apply_command(self._latched_command)
        self._step_physics(dt)
        self._step_count += 1
        self._t += dt
        return self._read_state(t=self._t)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._on_close()

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _reset_robot(self) -> None:
        """Reset the underlying physics state to the initial pose."""

    @abstractmethod
    def _read_state(self, t: float) -> RobotState:
        """Return a :class:`RobotState` reflecting the current physics state."""

    @abstractmethod
    def _apply_command(self, cmd: ControlCommand) -> None:
        """Translate ``cmd`` into a physics-level write (ctrl, qfrc_applied, …)."""

    @abstractmethod
    def _step_physics(self, dt: float) -> None:
        """Integrate the physics by ``dt`` seconds."""

    def _on_close(self) -> None:
        """Optional hook for releasing native resources. Default: no-op."""
