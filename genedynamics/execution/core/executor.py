"""
Main executor: orchestrates control loop for sim/real/shadow.

Runs plan_once or MPC, applies safety, drives state provider and control publisher.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from genedynamics.execution.core.contracts import (
    ActionPacket,
    ExecutionMode,
    PlanPacket,
    RobotState,
    SessionConfig,
)
from genedynamics.execution.core.clock import RuntimeClock
from genedynamics.execution.core.safety import SafetyGuard
from genedynamics.execution.bridges.planner_bridge import PlannerBridge
from genedynamics.execution.logging.telemetry import TelemetryLogger
from genedynamics.execution.logging.episode_writer import EpisodeWriter
from genedynamics.execution.providers.sim_state_provider import SimStateProvider
from genedynamics.execution.publishers.sim_control_publisher import SimControlPublisher


class Executor:
    """
    Main execution orchestrator.

    Supports sim_only, shadow, replay modes.
    """

    def __init__(
        self,
        state_provider: SimStateProvider,
        control_publisher: SimControlPublisher,
        planner_bridge: PlannerBridge,
        safety_guard: SafetyGuard,
        config: SessionConfig,
        telemetry: Optional[TelemetryLogger] = None,
        episode_writer: Optional[EpisodeWriter] = None,
    ):
        self.state_provider = state_provider
        self.control_publisher = control_publisher
        self.planner_bridge = planner_bridge
        self.safety_guard = safety_guard
        self.config = config
        self.telemetry = telemetry or TelemetryLogger(enabled=config.record)
        self.episode_writer = episode_writer

        self._clock = RuntimeClock(
            control_dt=config.get_control_dt(),
            real_time_factor=config.real_time_factor,
            sync_mode=config.sync_mode,
        )
        self._plan_cache: Optional[PlanPacket] = None
        self._plan_step = 0
        self._running = False
        self._rng = None

    def initialize(self, session_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Initialize executor."""
        if session_cfg:
            self._rng = session_cfg.get("rng")
        self._clock.start()
        self._running = True
        self.telemetry.start_episode(meta={"config": self.config.extra, **self.config.tags})

    def run_episode(
        self,
        plan_mode: str = "plan_once",
        max_steps: int = 150,
        initial_state: Optional[np.ndarray] = None,
        rng: Any = None,
    ) -> Dict[str, Any]:
        """
        Run one episode.

        plan_mode: "plan_once" | "mpc"
        """
        if rng is None:
            rng = self._rng
        env = self.control_publisher.env

        if initial_state is None:
            if env is not None and hasattr(env, "reset"):
                x0, _ = env.reset(rng=rng)
            elif hasattr(self.state_provider, "get_state"):
                s0 = self.state_provider.get_state()
                if s0 is not None:
                    x0 = s0.to_flat()
                else:
                    raise ValueError("initial_state required when state_provider returns None")
            else:
                raise ValueError("initial_state required when env has no reset")
        else:
            x0 = np.asarray(initial_state, dtype=np.float32)

        nq = getattr(self.state_provider, "nq", x0.size // 2)
        nv = getattr(self.state_provider, "nv", x0.size - nq)
        state = RobotState.from_flat(x0, nq, timestamp=0.0, source="sim")
        if hasattr(self.state_provider, "set_state"):
            self.state_provider.set_state(state)

        states: List[np.ndarray] = [x0.copy()]
        actions: List[np.ndarray] = []
        costs: List[float] = []
        plan_times: List[float] = []

        self._plan_cache = None
        self._plan_step = 0
        self._prev_action = None
        extra = getattr(self.config, "extra", {}) or {}
        action_smooth_alpha = float(extra.get("action_smooth_alpha", 0.0))
        t_sim = 0.0
        dt = self.config.get_control_dt()

        for t in range(max_steps):
            state = self.state_provider.get_state()
            if state is None:
                break

            x = state.to_flat()

            violation = self.safety_guard.check_state_violation(state)
            if violation is not None and violation.severity == "critical":
                fallback = self.safety_guard.fallback_policy(violation)
                action = fallback.action
                self.telemetry.log_event("safety_fallback", reason=fallback.reason)
            else:
                if plan_mode == "plan_once":
                    if self._plan_cache is None or self._plan_step >= len(self._plan_cache.actions):
                        t0 = time.perf_counter()
                        plan_horizon = min(
                            self.planner_bridge.horizon,
                            max_steps - t,
                        )
                        self._plan_cache = self.planner_bridge.plan_once(
                            state, {"rng": rng, "horizon": plan_horizon}
                        )
                        plan_times.append((time.perf_counter() - t0) * 1000.0)
                        self._plan_step = 0
                    action = self._plan_cache.get_action_at(self._plan_step)
                    self._plan_step += 1
                else:
                    t0 = time.perf_counter()
                    packet = self.planner_bridge.plan_mpc_step(state, {"rng": rng})
                    plan_times.append((time.perf_counter() - t0) * 1000.0)
                    action = packet.action

                if action is None:
                    action = np.zeros(getattr(env, "act_dim", 4), dtype=np.float32)
                action, _ = self.safety_guard.filter_action(action, state)
                # Action smoothing: blend with previous to reduce jumping at replan boundaries
                if action_smooth_alpha > 0 and self._prev_action is not None:
                    action = (
                        action_smooth_alpha * np.asarray(self._prev_action, dtype=np.float32)
                        + (1.0 - action_smooth_alpha) * np.asarray(action, dtype=np.float32)
                    )
            self._prev_action = action.copy() if action is not None else None

            self.control_publisher.publish(action, "position")

            if hasattr(self.control_publisher, "apply_to_env"):
                next_x = self.control_publisher.apply_to_env(x, action, t)
                if hasattr(self.state_provider, "set_state_from_flat"):
                    self.state_provider.set_state_from_flat(next_x, timestamp=t_sim + dt)
                state_for_log = next_x
            else:
                state_for_log = x

            states.append(state_for_log.copy())
            actions.append(action.copy())
            cost = getattr(env, "cost", lambda s: 0.0)(state_for_log) if env is not None else 0.0
            costs.append(float(cost))

            plan_lat = plan_times[-1] if plan_times else 0.0
            self.telemetry.log_state_action(state_for_log, action, t_sim, plan_latency_ms=plan_lat)

            t_sim += dt
            if self.config.execution_mode == ExecutionMode.REAL_ONLY:
                self._clock.sleep_until_next_step(plan_timestamp=t_sim)
            elif self.config.real_time_factor > 0 and self.config.execution_mode == ExecutionMode.SIM_ONLY:
                self._clock.sleep_until_next_step(plan_timestamp=t_sim)

        result = {
            "states": states,
            "actions": actions,
            "costs": costs,
            "plan_times_ms": plan_times,
            "steps": len(actions),
        }
        if self.telemetry.enabled:
            result["telemetry"] = self.telemetry.flush_episode()
            if self.episode_writer and self.config.record:
                extra = getattr(self.config, "extra", {}) or {}
                env_params = extra.get("env_params", {}) if isinstance(extra, dict) else {}
                env_name = (env_params.get("env_name") or extra.get("env_name") if isinstance(extra, dict) else None)
                if not env_name and hasattr(env, "model"):
                    env_name = "quadruped_go2_brax" if str(env.model) == "go2" else "humanoid_run_brax"
                meta = {"plan_mode": plan_mode, "max_steps": max_steps}
                if env_name:
                    meta["env_name"] = env_name
                ep_path = self.episode_writer.write_episode(
                    np.stack(states),
                    np.stack(actions) if actions else np.zeros((0, getattr(env, "act_dim", getattr(self.control_publisher, "act_dim", 4)))),
                    meta=meta,
                    telemetry=result.get("telemetry"),
                )
                result["episode_path"] = str(ep_path)
        return result

    def shutdown(self, reason: str = "normal") -> None:
        """Shutdown executor."""
        self._running = False
        self.control_publisher.emergency_stop()
