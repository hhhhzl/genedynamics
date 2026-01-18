from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import numpy as np


Context = Dict[str, Any]


@dataclass(frozen=True)
class D3ILTaskConfig:
    """
    Common config shared by D3IL tasks.

    Each task spec can extend this (or define its own) but we keep `render` here
    since nearly all MuJoCo tasks need it.
    """

    render: bool = False


@runtime_checkable
class D3ILTaskSpec(Protocol):
    """
    Spec for adapting a D3IL (classic gym) task into EnerDynamics-style env calls.

    The goal is: new tasks should be implemented by adding a new spec class,
    not by duplicating wrapper env logic.
    """

    # Static properties for logging / planner compatibility
    dt: float
    horizon: int
    state_dim: int
    act_dim: int
    control_limit: float
    target: np.ndarray

    def make_env(self) -> Any:
        """Construct the underlying D3IL environment object (gym.Env-like)."""

    def start_env(self, env: Any) -> None:
        """Start underlying D3IL env (D3IL requires explicit `env.start()`)."""

    def reset_to_state(self, env: Any, rng: Optional[Any] = None, **kwargs: Any) -> Tuple[np.ndarray, Context, Dict[str, Any]]:
        """
        Reset the env and return (state, ctx, info).

        ctx: task-specific context (e.g., fixed z, desired setpoints, etc.).
        """

    def action_to_env_action(self, env: Any, action: np.ndarray, ctx: Context) -> Tuple[np.ndarray, Context]:
        """Convert planner action to env action; may update ctx (e.g. desired setpoint)."""

    def step_to_state(self, env: Any, env_action: np.ndarray, ctx: Context) -> Tuple[np.ndarray, float, bool, Context, Dict[str, Any]]:
        """Step env and return (next_state, cost, done, ctx, info)."""

    def approx_transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Optional deterministic approximation (used only for planner-style interfaces)."""


