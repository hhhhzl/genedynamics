"""Semantic source queries; target joint coordinates never enter this API.

Construction metadata and records are framework neutral. Numerical providers
own IK, velocity matching, collision checks and the opaque Panda virtual state.
"""
from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Any, NamedTuple, Protocol

from genedynamics.registry_base import BaseRegistry
from .contracts import ExecutionSpec, MechanismSpec, ObstacleTransferConfig, SkillSpec


class SupportReason(IntEnum):
    SUPPORTED = 0
    INVALID_SEMANTICS = 1
    UNSUPPORTED_PHASE = 2
    POSE_UNREACHABLE = 3
    COLLISION = 4
    VELOCITY_UNREACHABLE = 5


@dataclass(frozen=True)
class SourceQuerySpec:
    iterations: int = 96
    damping: float = .002
    joint_step_limit: float = .12
    position_tolerance: float = .001
    orientation_tolerance: float = .01
    twist_tolerance: float = .005

    def __post_init__(self):
        if not isinstance(self.iterations, int) or self.iterations < 1:
            raise ValueError("source IK iterations must be a positive integer")
        for name in ("damping", "joint_step_limit", "position_tolerance",
                     "orientation_tolerance", "twist_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


class TaskSemantics(NamedTuple):
    tcp_position: Any  # [B,3], task/world frame, m
    tcp_rotation: Any  # [B,3,3], proper rotation, columns are tool axes
    tcp_twist: Any     # [B,6], linear then spatial angular velocity, m/s + rad/s
    phase: Any         # [B], obstacle transfer supports free-space phase 0 only


class SourceQueryResult(NamedTuple):
    semantics: TaskSemantics
    virtual_state: Any  # ExecutionState in the SOURCE backend
    support: Any        # [B], caller must stop/count failures before using prior
    reason: Any         # [B], SupportReason
    position_error: Any
    orientation_error: Any
    twist_error: Any
    clearance_lower_bound: Any


class SourceQueryBackend(Protocol):
    def build(self, semantics, request, *, previous=None) -> SourceQueryResult: ...
    def observe(self, target, state) -> TaskSemantics: ...
    def synchronize(self, value) -> None: ...
    def describe(self) -> dict: ...


def validate_semantics(semantics, batch):
    # Shape inspection only: never materialize device tensors in the frontend.
    for value, shape in zip(semantics, ((batch, 3), (batch, 3, 3), (batch, 6), (batch,))):
        if tuple(value.shape) != shape:
            raise ValueError(f"source query semantic shape must be {shape}")


source_query_registry = BaseRegistry[SourceQueryBackend]("genedynamics.cross_embodiment.source_queries")


def _jax(**kwargs):
    from .backends.jax.source_query import JaxSourceQueryBackend
    return JaxSourceQueryBackend(**kwargs)


def _numpy(**kwargs):
    from genedynamics.envs.domains.manipulation.cross_embodiment.backends.source_query import MujocoSourceQueryBackend
    return MujocoSourceQueryBackend(**kwargs)


source_query_registry.register("jax", SourceQueryBackend, factory=_jax)
source_query_registry.register("numpy", SourceQueryBackend, factory=_numpy)


class SourceQueryAdapter:
    """Build valid Panda state from shared semantics, with explicit support.

    The source has its own nominal mechanism and freshly initialized controller
    memory. No target wrench, integral or delay history is copied. Only phase 0
    is supported; contact-history reconstruction needs its own later adapter.
    """
    def __init__(self, *, execution=None, task=None, skill=None, query=None):
        self.execution = execution or ExecutionSpec()
        self.task = task or ObstacleTransferConfig()
        if self.task.robot != "panda":
            raise ValueError("ObstacleTransfer source query requires a Panda source task")
        self.skill, self.query = skill or SkillSpec(), query or SourceQuerySpec()
        self._impl = source_query_registry.create(self.execution.backend, execution=self.execution, task=self.task,
                                                 skill=self.skill, query=self.query,
                                                 mechanism=MechanismSpec(payload_mass=self.task.nominal_payload_mass))

    def build(self, semantics, request, *, previous=None):
        validate_semantics(semantics, len(request.sample_ids))
        return self._impl.build(semantics, request, previous=previous)

    def observe(self, runner, state):
        """Read shared task state in one backend call, with no joint transfer."""
        if runner.execution.backend != self.execution.backend:
            raise ValueError("observe requires matching numerical backends")
        if tuple(runner.task.goal_position) != tuple(self.task.goal_position) or runner.task.goal_yaw != self.task.goal_yaw:
            raise ValueError("source and target require the same fixed workcell goal")
        return self._impl.observe(runner._impl, state)

    def synchronize(self, value):
        self._impl.synchronize(value)

    def describe(self):
        return self._impl.describe()
