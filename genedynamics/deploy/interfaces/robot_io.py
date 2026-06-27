"""Sim/real-symmetric robot IO contract.

A :class:`RobotIO` is the only object in the deploy pipeline that knows
whether the robot is being simulated or driven in hardware. Everything else
(:class:`~genedynamics.deploy.interfaces.controller.Controller`,
:class:`~genedynamics.deploy.interfaces.follower.TrajectoryFollower`,
:class:`~genedynamics.deploy.interfaces.safety.SafetyFilter`,
observers, the pipeline loop) consumes :class:`RobotState` and emits
:class:`ControlCommand` and never branches on the IO type.

Concrete implementations live under ``deploy/io/``:

* ``MujocoRobotIO``  — wraps :mod:`genedynamics.envs.domains.humanoid.physics`
* ``MjxRobotIO``     — wraps :mod:`genedynamics.envs.domains.humanoid.mjx` (default)
* ``BraxRobotIO``    — wraps :mod:`genedynamics.envs.domains.humanoid.brax`
* ``UnitreeG1RobotIO`` — real Unitree G1 SDK
* ``StubRobotIO``    — no-op for tests
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Tuple, runtime_checkable

from genedynamics.deploy.interfaces.messages import ControlCommand, RobotState

__all__ = ["RobotIO"]


@runtime_checkable
class RobotIO(Protocol):
    """Symmetric IO interface for sim and real robots.

    Implementations expose:

    * ``spec``: an opaque robot spec (typically a
      :class:`~genedynamics.robots.g1.G1RobotSpec` or analogous structure).
      Controllers consume it for joint indexing, limits, and FK helpers.
    * ``physics_backend``: a string tag identifying the simulator
      (``"mujoco"`` / ``"mjx"`` / ``"brax"`` / ``"isaac"``) or ``None`` for
      real hardware. Used by the pipeline to validate runtime compatibility.
    * ``array_runtime``: a string tag identifying the array library this IO
      produces (``"numpy"`` / ``"jax"`` / ``"torch"``). The pipeline asserts
      that the active :class:`Controller` uses a compatible runtime.
    * ``accepts``: an iterable of supported :class:`ControlCommand.kind`
      values. Pipelines verify the active controller's outputs are accepted.
    """

    spec: Any
    physics_backend: Optional[str]
    array_runtime: str
    accepts: Tuple[str, ...]

    def reset(self) -> RobotState:
        """Reset the robot to its initial state and return the first observation."""
        ...

    def get_state(self) -> RobotState:
        """Return the latest measured (or simulated) state without advancing time."""
        ...

    def send_control(self, cmd: ControlCommand) -> None:
        """Latch a command for the next call to :meth:`step`.

        Real implementations may also dispatch immediately if the underlying
        protocol is asynchronous; in that case ``step`` becomes a no-op.
        """
        ...

    def step(self, dt: float) -> RobotState:
        """Advance simulation by ``dt`` seconds and return the new state.

        On real hardware this is a no-op other than waiting for the next
        sensor packet; ``dt`` is interpreted as the desired control period
        and used by an internal rate limiter when present.
        """
        ...

    def close(self) -> None:
        """Release any external resources (sockets, SDK handles, render windows)."""
        ...
