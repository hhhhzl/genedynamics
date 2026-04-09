"""Deploy-side execution tasks.

This package owns the tiny ``ExecutionTask`` protocol that the runner uses
to ask "is the episode done?" / "did we succeed?" / "what should the
follower trace?". It deliberately does **not** redefine task layout — that
lives in :mod:`genedynamics.tasks.*` (one ``TaskSpec`` per robot family).
What lives here is the per-step glue that bridges the static spec to the
running pipeline.

Implementations:

* :class:`BaseExecutionTask` — no-op base; "never done"
* :class:`CorridorFollowTask` — consumes a 14D corridor plan, computes
  per-step progress against the target endpoint, signals ``done`` when the
  pelvis is within ``goal_tolerance_m`` (or ``max_steps`` is reached)
* :class:`TeleopTask` — live teleoperation monitor; signals ``done`` on
  E-stop, fall detection, or ``max_steps``
"""

from genedynamics.deploy.tasks.base import BaseExecutionTask, ExecutionTask
from genedynamics.deploy.tasks.corridor_follow import CorridorFollowTask
from genedynamics.deploy.tasks.teleop_task import TeleopTask

__all__ = [
    "BaseExecutionTask",
    "CorridorFollowTask",
    "ExecutionTask",
    "TeleopTask",
]
