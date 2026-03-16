"""
Execution layer contracts and data types.

Defines protocol interfaces and dataclasses for state, control, planning,
and safety. All types are designed for serialization and replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np


@dataclass(frozen=True)
class RobotState:
    """Canonical robot state for execution layer."""

    qpos: np.ndarray  # position / generalized coordinates
    qvel: np.ndarray  # velocity / generalized velocities
    timestamp: float = 0.0
    source: str = "unknown"  # "sim" | "real" | "localization"

    def to_flat(self) -> np.ndarray:
        """Flatten to [qpos; qvel] for planning."""
        return np.concatenate([np.asarray(self.qpos), np.asarray(self.qvel)]).astype(np.float32)

    @classmethod
    def from_flat(cls, flat: np.ndarray, nq: int, timestamp: float = 0.0, source: str = "unknown") -> RobotState:
        """Construct from flat state [qpos; qvel]."""
        flat = np.asarray(flat, dtype=np.float32)
        nv = flat.size - nq
        return cls(
            qpos=flat[:nq].copy(),
            qvel=flat[nq : nq + nv].copy(),
            timestamp=timestamp,
            source=source,
        )

    def __len__(self) -> int:
        return self.qpos.size + self.qvel.size


@dataclass
class ActionPacket:
    """Single control action with metadata."""

    action: np.ndarray
    mode: str = "position"  # "position" | "torque" | "hybrid"
    timestamp: float = 0.0
    ttl_ms: float = 0.0  # time-to-live for real-time use


@dataclass
class PlanPacket:
    """Planning output: trajectory and metadata."""

    states: List[np.ndarray]
    actions: List[np.ndarray]
    horizon: int
    planning_time_ms: float = 0.0
    info: Dict[str, Any] = field(default_factory=dict)

    def get_action_at(self, step: int) -> Optional[np.ndarray]:
        """Get action at step index."""

        if 0 <= step < len(self.actions):
            return np.asarray(self.actions[step], dtype=np.float32)
        return None


class HealthStatus(Enum):
    """Health status for state/control providers."""

    OK = auto()
    DEGRADED = auto()
    STALE = auto()
    FAILED = auto()


@dataclass
class ViolationEvent:
    """Safety violation event."""

    kind: str  # "state_bounds" | "action_limits" | "collision" | "timeout"
    message: str
    severity: str = "warn"  # "warn" | "critical"
    state: Optional[np.ndarray] = None
    action: Optional[np.ndarray] = None
    timestamp: float = 0.0


@dataclass
class FallbackAction:
    """Fallback action when safety triggers."""

    action: np.ndarray
    reason: str
    original_action: Optional[np.ndarray] = None


class ExecutionMode(Enum):
    """Execution mode."""

    SIM_ONLY = "sim_only"
    REAL_ONLY = "real_only"
    SHADOW = "shadow"
    REPLAY = "replay"
    PLAN_ONLY = "plan_only"


@dataclass
class SessionConfig:
    """Session configuration for execution."""

    execution_mode: ExecutionMode = ExecutionMode.SIM_ONLY
    control_rate_hz: float = 20.0
    plan_rate_hz: float = 10.0
    sim_dt: float = 0.0  # 0 = infer from control_rate
    real_time_factor: float = 1.0
    sync_mode: bool = True
    plan_timeout_ms: float = 500.0
    max_missed_cycles: int = 3
    fallback: str = "hold"  # "hold" | "stand" | "stop"
    action_clip_min: Optional[np.ndarray] = None
    action_clip_max: Optional[np.ndarray] = None
    state_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    record: bool = True
    episode_dir: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def get_control_dt(self) -> float:
        """Control period in seconds."""
        return 1.0 / max(1e-6, self.control_rate_hz)

    def get_plan_dt(self) -> float:
        """Planning period in seconds."""
        return 1.0 / max(1e-6, self.plan_rate_hz)


class StateProvider(Protocol):
    """Protocol for state providers."""

    def get_state(self) -> Optional[RobotState]:
        """Get current state. Returns None if unavailable."""
        ...

    def get_timestamp(self) -> float:
        """Get last update timestamp."""
        ...

    def health(self) -> HealthStatus:
        """Health status."""
        ...


class ControlPublisher(Protocol):
    """Protocol for control publishers."""

    def publish(self, action: np.ndarray, mode: str, ttl_ms: float = 0.0) -> None:
        """Publish control action."""
        ...

    def set_safe_posture(self, posture: np.ndarray) -> None:
        """Set safe posture (e.g. hold, stand)."""
        ...

    def emergency_stop(self) -> None:
        """Emergency stop."""
        ...


class PlannerBridge(Protocol):
    """Protocol for planner bridge."""

    def plan_once(self, state: RobotState, context: Dict[str, Any]) -> PlanPacket:
        """Plan full trajectory from state."""
        ...

    def plan_mpc_step(self, state: RobotState, context: Dict[str, Any]) -> ActionPacket:
        """Plan single next action (MPC step)."""
        ...
