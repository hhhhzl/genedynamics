"""
Goal-direction task bias for 2D navigation.

Provides a normalized direction from current actions toward target
actions, decayed by (1 - hardness) so that the bias is strongest in
the early diffusion steps and vanishes as constraints harden.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from genedynamics.genemetry.base import TaskDirectionProvider
from genedynamics.genemetry.registry import register_genemetry


@register_genemetry("modulation", "goal_direction", "jax")
class GoalDirectionJax(TaskDirectionProvider):
    """Goal-direction provider (JAX).

    Parameters
    ----------
    alpha_task : float
        Base strength of the task direction bias.
    """

    def __init__(self, *, alpha_task: float = 0.15, **kwargs: Any) -> None:
        self._alpha_task = float(alpha_task)

    def direction(
        self,
        current_actions: jnp.ndarray,
        target_actions: jnp.ndarray,
        step_k: jnp.ndarray,
        hardness: jnp.ndarray,
    ) -> jnp.ndarray:
        diff = target_actions - current_actions
        norm = jnp.linalg.norm(diff)
        d = diff / jnp.maximum(norm, jnp.asarray(1e-6, dtype=jnp.float32))
        # Decay with hardness: strong early, weak late.
        weight = self._alpha_task * (1.0 - hardness)
        return weight * d
