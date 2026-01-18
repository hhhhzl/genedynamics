"""
ConstraintFilter abstraction for action-space filtering.

This is intentionally separate from the "operator" QP layer:
- Operators expect convex constraints already built (ConvexConstraint).
- Filters can *internally* use convexifiers/operators, or use closed-form updates.

MDOC will call a ConstraintFilter once per diffusion step to filter sampled actions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class ConstraintFilter(ABC):
    """
    Filter actions in action space given current schedule state/params.

    Conventions:
    - actions shape is either (H, act_dim) or (B, H, act_dim)
    - returns same shape as input actions
    """

    @abstractmethod
    def apply_actions(
        self,
        x0: Any,
        actions: Any,
        *,
        env: Any,
        obstacles: Any = None,
        # NOTE: JAX-jitted code cannot create Python dataclasses from tracers.
        # We therefore allow these to be arbitrary pytrees (e.g., dicts of jnp scalars).
        schedule_state: Optional[Any] = None,
        schedule_params: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Filter actions (single sequence or batch)."""
        raise NotImplementedError

    def apply_actions_batch(
        self,
        x0: Any,
        actions_batch: Any,
        *,
        env: Any,
        obstacles: Any = None,
        schedule_state: Optional[Any] = None,
        schedule_params: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Default batch implementation: call apply_actions directly.
        Subclasses can override for better performance.
        """
        return self.apply_actions(
            x0,
            actions_batch,
            env=env,
            obstacles=obstacles,
            schedule_state=schedule_state,
            schedule_params=schedule_params,
            **kwargs,
        )


