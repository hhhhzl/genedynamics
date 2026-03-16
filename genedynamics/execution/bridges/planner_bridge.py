"""
Planner bridge: adapts planner/solver to execution layer protocol.

Wraps 2GO, MBD, etc. to provide plan_once and plan_mpc_step.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from genedynamics.execution.core.contracts import ActionPacket, PlanPacket, RobotState


class PlannerBridge:
    """
    Bridge from planner (solver) to execution contracts.

    Planner must have:
    - solve(x0, horizon=..., rng_key=...) -> object with .actions, .states, .info
    """

    def __init__(
        self,
        planner: Any,
        horizon: int,
        rng: Any = None,
        plan_mode: str = "plan_once",  # "plan_once" | "mpc"
    ):
        self.planner = planner
        self.horizon = horizon
        self._rng = rng
        self.plan_mode = plan_mode
        self._plan_cache: Optional[PlanPacket] = None
        self._cache_step = 0

    def plan_once(self, state: RobotState, context: Dict[str, Any]) -> PlanPacket:
        """Plan full trajectory from state."""
        x0 = state.to_flat()
        rng = context.get("rng", self._rng)
        horizon = context.get("horizon", self.horizon)

        t0 = time.perf_counter()
        result = self.planner.solve(x0, horizon=horizon, rng_key=rng)
        t_ms = (time.perf_counter() - t0) * 1000.0

        actions = getattr(result, "actions", None)
        states = getattr(result, "states", None)
        if actions is None:
            actions = []
        if states is None:
            states = []
        info = getattr(result, "info", None) or {}

        if isinstance(actions, np.ndarray):
            actions = [np.asarray(actions[i], dtype=np.float32) for i in range(actions.shape[0])]
        if isinstance(states, np.ndarray):
            states = [np.asarray(states[i], dtype=np.float32) for i in range(states.shape[0])]

        packet = PlanPacket(
            states=list(states),
            actions=list(actions),
            horizon=horizon,
            planning_time_ms=t_ms,
            info=dict(info),
        )
        self._plan_cache = packet
        self._cache_step = 0
        return packet

    def plan_mpc_step(self, state: RobotState, context: Dict[str, Any]) -> ActionPacket:
        """Plan single next action (MPC step)."""
        if self.plan_mode == "plan_once" and self._plan_cache is not None:
            action = self._plan_cache.get_action_at(self._cache_step)
            if action is not None:
                self._cache_step += 1
                return ActionPacket(
                    action=action,
                    mode=context.get("mode", "position"),
                    timestamp=state.timestamp,
                )
        packet = self.plan_once(state, context)
        action = packet.get_action_at(0)
        if action is None:
            n = getattr(self.planner, "act_dim", 4)
            action = np.zeros(n, dtype=np.float32)
        return ActionPacket(
            action=action,
            mode=context.get("mode", "position"),
            timestamp=state.timestamp,
        )
