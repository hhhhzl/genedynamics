"""
Execution layer for sim/real deployment.

Provides unified execution across simulation and real hardware:
- Executor: main control loop orchestrator
- StateProvider: state source (sim, real, localization)
- ControlPublisher: control output (sim, real)
- SafetyGuard: action filtering, fallback, emergency
- PlannerBridge: planner I/O protocol
- TelemetryLogger / EpisodeWriter: observability and replay
"""

from genedynamics.execution.core.contracts import (
    RobotState,
    ActionPacket,
    PlanPacket,
    HealthStatus,
    ViolationEvent,
    FallbackAction,
    ExecutionMode,
    SessionConfig,
)
from genedynamics.execution.core.safety import SafetyGuard
from genedynamics.execution.core.executor import Executor
from genedynamics.execution.bridges.planner_bridge import PlannerBridge

__all__ = [
    "RobotState",
    "ActionPacket",
    "PlanPacket",
    "HealthStatus",
    "ViolationEvent",
    "FallbackAction",
    "ExecutionMode",
    "SessionConfig",
    "SafetyGuard",
    "Executor",
    "PlannerBridge",
]
