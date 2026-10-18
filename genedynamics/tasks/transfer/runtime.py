"""Backend-neutral construction metadata and fixed-structure execution records.

Array fields deliberately use Any: concrete tensor/state types belong to a
backend. NamedTuple records are also native JAX pytrees without importing JAX.
All public command/trace arrays have batch as their first axis; units are SI.
"""
from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Any, NamedTuple

SCHEMA_VERSION = 1
SKILL_DIM = 13
ENVIRONMENT_NAME = "obstacle_transfer"


class Status(IntEnum):
    ACTIVE = 0
    SUCCESS = 1
    COLLISION = 2
    JOINT_LIMIT = 3
    SPEED_LIMIT = 4
    TIMEOUT = 5
    NONFINITE = 6
    SOURCE_SUPPORT_FAILURE = 7


@dataclass(frozen=True)
class ExecutionSpec:
    backend: str = "jax"
    device: str = "cpu"
    physics_backend: str = "mjx"
    controller_backend: str = "builtin"
    execution_mode: str = "collect"
    dtype: str = "float32"
    microbatch_size: int | None = None

    def __post_init__(self):
        if self.execution_mode not in {"collect", "evaluate", "infer", "train"}:
            raise ValueError("unknown execution_mode")
        if self.dtype != "float32":
            raise ValueError("v1 command contract requires float32")
        if self.microbatch_size is not None and (
            isinstance(self.microbatch_size, bool) or not isinstance(self.microbatch_size, int)
            or self.microbatch_size < 1
        ):
            raise ValueError("microbatch_size must be a positive integer or None")


@dataclass(frozen=True)
class SkillSpec:
    horizon: int = 10
    linear_speed: float = 0.12
    angular_speed: float = 0.6
    log_stiffness_limit: float = 1.0
    linear_acceleration: float = 0.6
    angular_acceleration: float = 3.0
    log_stiffness_rate: float = 4.0
    reference_stiffness: tuple = (800., 800., 800., 60., 60., 60.)

    def __post_init__(self):
        values = (self.linear_speed, self.angular_speed, self.log_stiffness_limit,
                  self.linear_acceleration, self.angular_acceleration, self.log_stiffness_rate,
                  *self.reference_stiffness)
        if not isinstance(self.horizon, int) or self.horizon < 1 or len(self.reference_stiffness) != 6 or any(
            not math.isfinite(x) or x <= 0 for x in values
        ):
            raise ValueError("invalid skill bounds/stiffness")


@dataclass(frozen=True)
class ObstacleTransferConfig:
    robot: str = "panda"
    physics_dt: float = 0.002
    control_dt: float = 0.02
    start_position: tuple = (0.38, -0.12, 0.55)
    goal_position: tuple = (0.64, 0.08, 0.55)
    goal_yaw: float = 0.15
    reset_joint_noise: float = 0.002
    nominal_payload_mass: float = 0.5
    max_delay_steps: int = 4
    position_tolerance: float = 0.005
    orientation_tolerance: float = math.radians(3)
    hold_seconds: float = 0.3
    timeout_seconds: float = 10.
    joint_speed_limit: float = 2.5
    contact_margin: float = 0.005
    collision_tolerance: float = 1e-5

    def __post_init__(self):
        if self.robot not in {"panda", "xarm7", "ur5e"}:
            raise ValueError("unsupported ObstacleTransfer arm")
        for name in ("physics_dt", "control_dt", "nominal_payload_mass", "position_tolerance",
                     "orientation_tolerance", "hold_seconds", "timeout_seconds", "joint_speed_limit",
                     "contact_margin"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isclose(self.control_dt / self.physics_dt, self.substeps, abs_tol=1e-9):
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        if self.substeps < 1 or not isinstance(self.max_delay_steps, int) or self.max_delay_steps < 0:
            raise ValueError("invalid substeps/delay capacity")
        if not math.isfinite(self.reset_joint_noise) or self.reset_joint_noise < 0:
            raise ValueError("reset_joint_noise must be finite and nonnegative")
        if not math.isfinite(self.goal_yaw) or not math.isfinite(self.collision_tolerance) or self.collision_tolerance < 0:
            raise ValueError("invalid yaw/collision tolerance")
        for pos in (self.start_position, self.goal_position):
            if len(pos) != 3 or not all(math.isfinite(x) for x in pos):
                raise ValueError("task positions must be finite xyz triples")

    @property
    def substeps(self):
        return round(self.control_dt / self.physics_dt)

    @property
    def timeout_steps(self):
        return math.ceil(self.timeout_seconds / self.control_dt)

    @property
    def hold_steps(self):
        return math.ceil(self.hold_seconds / self.control_dt)


@dataclass(frozen=True)
class MechanismSpec:
    """Construction-time development mechanism; never a policy observation."""
    servo_gain: float = 1.
    delay_steps: int = 0
    payload_mass: float = 0.5

    def __post_init__(self):
        if any(not math.isfinite(x) or x <= 0 for x in (self.servo_gain, self.payload_mass)):
            raise ValueError("mechanism gain/mass must be finite and positive")
        if not isinstance(self.delay_steps, int) or self.delay_steps < 0:
            raise ValueError("delay_steps must be a nonnegative integer")


def mechanism_specs(mechanism):
    """Normalize construction truth; a tuple declares one mechanism per world.

    A single spec broadcasts to any batch size. A tuple has a fixed batch size
    and its order must match the execution state's world axis. This metadata is
    only used by execution backends and is never added to policy observations.
    """
    values = (mechanism,) if isinstance(mechanism, MechanismSpec) else mechanism
    if not isinstance(values, tuple) or not values or not all(isinstance(x, MechanismSpec) for x in values):
        raise TypeError("mechanism must be MechanismSpec or a nonempty tuple of MechanismSpec")
    return values


@dataclass(frozen=True)
class RandomRequest:
    seed: int
    sample_ids: tuple

    def __post_init__(self):
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be uint32")
        if not self.sample_ids or any(not isinstance(i, int) or not 0 <= i < 2**32 for i in self.sample_ids):
            raise ValueError("sample_ids must be nonempty uint32 identities")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample IDs must be unique within a batch")


class ControllerState(NamedTuple):
    reference_position: Any
    reference_rotation: Any
    command_buffer: Any
    last_command: Any
    integral_error: Any


class ExecutionState(NamedTuple):
    physics: Any  # opaque backend state; not part of the portable inference bundle
    controller: ControllerState
    sample_id: Any
    steps: Any
    hold_count: Any
    status: Any


class StepTrace(NamedTuple):
    tcp_position: Any
    tcp_rotation: Any
    position_error: Any
    orientation_error: Any
    clearance_lower_bound: Any  # censored at contact_margin; not an exact distance oracle
    max_joint_speed: Any
    saturation: Any
    command: Any
    actuator_command: Any
    status: Any
    valid: Any
    terminal: Any


class ChunkResult(NamedTuple):
    state: ExecutionState
    trace: StepTrace  # [B,L,...]; terminal true once, before any reset
