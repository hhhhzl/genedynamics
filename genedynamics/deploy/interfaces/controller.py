"""Controller protocol — the only contract between the loop and any policy.

Sport-mode, RL, WBC, PD and MPC controllers all conform to this single
interface. Switching from one to another is one config change; the rest of
the pipeline (:class:`RobotIO`, :class:`TrajectoryFollower`,
:class:`SafetyFilter`, observers) is untouched.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from genedynamics.deploy.interfaces.messages import ControlCommand, Intent, RobotState
from genedynamics.deploy.interfaces.robot_io import RobotIO

__all__ = ["Controller"]


@runtime_checkable
class Controller(Protocol):
    """Map ``(state, intent) → ControlCommand``.

    Attributes:
        spec: Robot spec the controller was constructed against. Typically a
            :class:`~genedynamics.robots.g1.G1RobotSpec`. Controllers should
            treat it as read-only.
        runtime: Tag for the array runtime backend (``"numpy"`` / ``"torch"``
            / ``"jax"`` / ``"rust"``). Pipelines verify it is compatible with
            the active :class:`RobotIO.array_runtime`. A WBC controller running
            ``osqp`` on host stays ``"numpy"`` even when paired with an MJX IO;
            the pipeline performs a single host↔device copy per step.
        produces: The :class:`ControlCommand.kind` values this controller may
            emit. The pipeline verifies the active IO accepts all of them.
    """

    spec: Any
    runtime: str
    produces: tuple[str, ...]

    def reset(self, io: RobotIO) -> None:
        """Initialize internal state for a fresh episode.

        ``io`` is provided so the controller can pre-fetch model handles or
        cache joint mappings, but it must not retain a reference past
        :meth:`act` boundaries.
        """
        ...

    def act(self, state: RobotState, intent: Intent) -> ControlCommand:
        """Compute one control command for the current step."""
        ...
