"""
No-op constraint filter (identity).
"""

from __future__ import annotations

from typing import Any, Optional

from enerdynamics.core.constraints.action_filters.base import ConstraintFilter


class NoOpConstraintFilter(ConstraintFilter):
    def apply_actions(
        self,
        x0: Any,
        actions: Any,
        *,
        env: Any,
        obstacles: Any = None,
        schedule_state: Optional[Any] = None,
        schedule_params: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        return actions


