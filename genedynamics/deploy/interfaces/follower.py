"""Trajectory follower protocol.

A :class:`TrajectoryFollower` translates a planner output (a 14D corridor
trajectory, a sequence of waypoints, a logged episode, …) into a per-step
:class:`Intent` consumed by the :class:`Controller`.

Followers are robot-agnostic: they speak in the language of base position,
yaw, height, linear velocity and yaw rate. Robot-specific lowering of the
intent into joint commands is the controller's job.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from genedynamics.deploy.interfaces.messages import Intent, RobotState

__all__ = ["TrajectoryFollower"]


@runtime_checkable
class TrajectoryFollower(Protocol):
    """Map ``(t, state) → Intent``.

    Followers are stateful — they typically own a parsed plan, an internal
    cursor, and a phase scheduler. They are reset at episode boundaries via
    :meth:`reset` with a fresh plan.

    Attributes:
        plan_dt: Native time step of the underlying plan in seconds, used by
            the pipeline to set up resampling / decimation.
        horizon: Total plan duration in seconds. ``None`` for open-ended
            followers (e.g. teleoperation).
    """

    plan_dt: float
    horizon: float | None

    def reset(self, plan: Any) -> None:
        """Bind to a new plan and rewind internal state."""
        ...

    def step(self, t: float, state: RobotState) -> Intent:
        """Sample the plan at time ``t`` and return the current intent.

        ``state`` is provided so closed-loop followers (e.g. those that wait
        for foot contact before advancing the phase) can react to the robot;
        open-loop followers may ignore it entirely.
        """
        ...

    def is_finished(self, t: float) -> bool:
        """Whether the plan has been fully consumed."""
        ...
