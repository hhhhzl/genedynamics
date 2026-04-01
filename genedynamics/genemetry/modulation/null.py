"""
Null (pass-through) task modulator.

Returns constraint geometry unchanged.  Used when no task-specific
modulation is needed (e.g. standard obstacle avoidance without
stepping-stone awareness).
"""

from typing import Any, Optional

from genedynamics.genemetry.base import TaskModulator


class NullModulator(TaskModulator):
    """Pass-through modulator -- returns geometry unmodified."""

    def modulate(
        self,
        raw_geometry: Any,
        probe_geometry: Optional[Any] = None,
        window_mask: Optional[Any] = None,
        alpha: float = 0.0,
    ) -> Any:
        return raw_geometry
