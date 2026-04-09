"""Safety filter protocol.

A :class:`SafetyFilter` sits between the controller and the robot IO,
projecting unsafe :class:`ControlCommand`\\ s onto the safe set defined by
joint / torque limits, self-collision avoidance, control barrier functions,
or any other constraint constructible from
:mod:`genedynamics.core.constraints`.

Multiple filters can be composed via a :class:`CompositeSafetyFilter`
implementation (lives under ``deploy/safety/``); they apply in declaration
order. The contract here is intentionally minimal: take a state + command,
return a (possibly modified) command + diagnostics.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from genedynamics.deploy.interfaces.messages import ControlCommand, RobotState

__all__ = ["SafetyFilter", "SafetyResult"]


from dataclasses import dataclass, field


@dataclass
class SafetyResult:
    """Outcome of a safety filter pass."""

    command: ControlCommand
    intervened: bool = False
    violations: dict[str, float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SafetyFilter(Protocol):
    """Project a candidate command onto the safe set.

    Attributes:
        spec: Robot spec the filter was constructed against.
        runtime: Tag for the array runtime backend, mirroring
            :class:`Controller.runtime`. Pipelines verify cross-component
            compatibility at startup.
    """

    spec: Any
    runtime: str

    def reset(self) -> None:
        """Reset internal filter state for a new episode."""
        ...

    def filter(self, state: RobotState, cmd: ControlCommand) -> SafetyResult:
        """Return a safe version of ``cmd`` along with intervention metadata."""
        ...
