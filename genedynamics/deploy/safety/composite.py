"""Composite safety filter — apply a list of filters in declaration order.

The :class:`CompositeSafetyFilter` is itself a :class:`SafetyFilter`, so it
can be plugged into the pipeline anywhere a single filter is expected. Each
inner filter receives the **modified** command from the previous filter,
and any of them can flag intervention. Violations are merged into a single
namespaced dict so observers can see exactly which filter triggered.

The composite is also responsible for fanning out :meth:`reset` calls so
filters with episode-level state (e.g. CBF integrators) get the lifecycle
they need.
"""

from __future__ import annotations

from typing import Any, Iterable, List

from genedynamics.deploy.interfaces.messages import ControlCommand, RobotState
from genedynamics.deploy.interfaces.safety import SafetyFilter, SafetyResult

__all__ = ["CompositeSafetyFilter"]


class CompositeSafetyFilter:
    """Chain of :class:`SafetyFilter`\\ s applied in order.

    Args:
        spec: Robot spec — must match the spec each inner filter was built
            against. Stored on the composite so the pipeline can validate
            consistency.
        filters: Iterable of filters. Order matters: each filter sees the
            previous filter's output.
    """

    runtime: str = "numpy"

    def __init__(self, spec: Any, filters: Iterable[SafetyFilter]) -> None:
        self.spec = spec
        self.filters: List[SafetyFilter] = list(filters)

    def reset(self) -> None:
        for f in self.filters:
            f.reset()

    def filter(self, state: RobotState, cmd: ControlCommand) -> SafetyResult:
        current_cmd = cmd
        intervened_any = False
        violations: dict[str, float] = {}
        extras: dict[str, Any] = {}
        for f in self.filters:
            result = f.filter(state, current_cmd)
            current_cmd = result.command
            if result.intervened:
                intervened_any = True
                name = type(f).__name__
                for k, v in result.violations.items():
                    violations[f"{name}.{k}"] = v
                if result.extras:
                    extras[name] = result.extras
        return SafetyResult(
            command=current_cmd,
            intervened=intervened_any,
            violations=violations,
            extras=extras,
        )
