"""Base scaffolding for :class:`SafetyFilter` implementations.

Concrete filters subclass :class:`BaseSafetyFilter` to inherit:

* a default no-op :meth:`reset` (no per-episode state),
* the ``spec`` / ``runtime`` attribute fields the protocol requires,
* a small helper :meth:`_passthrough` for filters that decide their input
  was already safe.

A filter that needs episode-level state (e.g. CBF integrator memory) just
overrides :meth:`reset`.
"""

from __future__ import annotations

from typing import Any

from genedynamics.deploy.interfaces.messages import ControlCommand
from genedynamics.deploy.interfaces.safety import SafetyResult

__all__ = ["BaseSafetyFilter"]


class BaseSafetyFilter:
    """Common base class providing the protocol-required attributes."""

    runtime: str = "numpy"

    def __init__(self, spec: Any) -> None:
        self.spec = spec

    def reset(self) -> None:
        """Default no-op reset. Override if the filter has internal state."""

    def _passthrough(self, cmd: ControlCommand) -> SafetyResult:
        """Return ``cmd`` unchanged with ``intervened=False``."""
        return SafetyResult(command=cmd, intervened=False)
