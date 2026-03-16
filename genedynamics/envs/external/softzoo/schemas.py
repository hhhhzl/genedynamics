"""
Unified schemas for SoftZoo integration.

Defines request/result types and specs for industrial-grade extensibility.
All types are dataclasses for serialization and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# -----------------------------------------------------------------------------
# Enums for type safety and extensibility
# -----------------------------------------------------------------------------


class SoftZooEnvType(str, Enum):
    """Environment type (land, aquatic, manipulation)."""

    LAND = "land_environment"
    AQUATIC = "aquatic_environment"
    MANIPULATION = "manipulation_environment"
    DUMMY = "dummy_env"


class SoftZooDesignerType(str, Enum):
    """Design representation type."""

    ANNOTATED_PCD = "annotated_pcd"
    VOXEL = "voxel"
    SDF_BASIS = "sdf_basis"
    DIFF_CPPN = "diff_cppn"
    WASS_BARYCENTER = "wass_barycenter"
    DUMMY = "dummy"


class SoftZooObjective(str, Enum):
    """Task objective / reward type."""

    MOVE_FORWARD = "move_forward"
    TRAJECTORY_FOLLOWING = "trajectory_following"
    MOVE_IN_CIRCLES = "move_in_circles"
    DUMMY = "dummy"


# -----------------------------------------------------------------------------
# Spec types (immutable configuration)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SoftZooMorphologySpec:
    """
    Morphology specification for a soft robot design.

    Attributes:
        designer_type: Design representation
        pcd_path: Path to annotated PCD (for annotated_pcd)
        pcd_name: Short name (e.g. Caterpillar) for path resolution
        passive_geometry_mul: Passive geometry multiplier
        voxel_size: For voxel-based (optional)
    """

    designer_type: SoftZooDesignerType = SoftZooDesignerType.ANNOTATED_PCD
    pcd_path: Optional[str] = None
    pcd_name: str = "Caterpillar"
    passive_geometry_mul: float = 0.2
    voxel_size: Optional[Tuple[int, int, int]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SoftZooControllerSpec:
    """
    Controller specification.

    Attributes:
        n_actuators: Number of actuators
        actuation_omega: (min, max) for actuation frequency
        lr: Learning rate for differentiable controller
        action_space: particle_v | actuator_v
    """

    n_actuators: int = 10
    actuation_omega: Tuple[float, float] = (20.0, 100.0)
    lr: float = 0.1
    action_space: str = "particle_v"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SoftZooModeSpec:
    """
    Contact/friction mode specification.

    Attributes:
        mode_id: Integer mode index
        name: Human-readable name
        friction: Friction coefficient
        damping: Optional damping
        terrain_variant: Optional terrain override
        log_prior: log p(mode)
    """

    mode_id: int
    name: str
    friction: float = 0.5
    damping: Optional[float] = None
    terrain_variant: Optional[str] = None
    log_prior: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SoftZooFidelitySpec:
    """
    Fidelity level specification.

    Attributes:
        level: 0=coarse, 1=medium, 2=fine
        max_substeps: Max simulation substeps
        max_substeps_local: Local substeps
        quality: Grid quality
        n_frames: Frames per episode
    """

    level: int
    max_substeps: int = 3500
    max_substeps_local: int = 20
    quality: float = 1.0
    n_frames: int = 200
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SoftZooRobotSpec:
    """Combined robot spec (morphology + controller)."""

    morphology: SoftZooMorphologySpec
    controller: SoftZooControllerSpec


@dataclass(frozen=True)
class SoftZooTaskSpec:
    """
    Full task specification for a SoftZoo task.

    Attributes:
        task_id: Unique task identifier
        env_type: Environment type
        env_config_file: YAML config name (e.g. ground.yaml)
        objective: Task objective
        robot: Robot spec
        modes: List of mode specs
        fidelity_levels: List of fidelity specs
        max_steps: Max episode steps
        forward_direction: [x,y,z] for locomotion
    """

    task_id: str
    env_type: SoftZooEnvType = SoftZooEnvType.LAND
    env_config_file: str = "ground.yaml"
    objective: SoftZooObjective = SoftZooObjective.MOVE_FORWARD
    robot: SoftZooRobotSpec = field(
        default_factory=lambda: SoftZooRobotSpec(
            morphology=SoftZooMorphologySpec(),
            controller=SoftZooControllerSpec(),
        )
    )
    modes: Tuple[SoftZooModeSpec, ...] = ()
    fidelity_levels: Tuple[SoftZooFidelitySpec, ...] = ()
    max_steps: int = 200
    forward_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    extra: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Request / Result types
# -----------------------------------------------------------------------------


@dataclass
class SoftZooRolloutRequest:
    """
    Request for a batch of rollouts.

    Attributes:
        task_id: Task identifier
        morphology_params: Design parameters x (flattened or structured)
        controller_params: Controller parameters phi
        mode_id: Contact mode index
        fidelity_level: Fidelity level
        seed: Random seed
        num_repeats: Number of repeats per sample (for variance reduction)
        render: Whether to render
        record: Whether to record trajectory
    """

    task_id: str
    morphology_params: np.ndarray
    controller_params: np.ndarray
    mode_id: int = 0
    fidelity_level: int = 2
    seed: int = 0
    num_repeats: int = 1
    render: bool = False
    record: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SoftZooRolloutResult:
    """
    Result from a single rollout.

    Attributes:
        return_: Episode return (sum of rewards)
        success: Whether task succeeded
        num_steps: Steps taken
        final_com: Final center-of-mass position
        trajectory: Optional state/action trajectory
        failure_code: Optional failure code
        wall_time: Wall-clock time in seconds
    """

    return_: float
    success: bool
    num_steps: int
    final_com: np.ndarray
    trajectory: Optional[Dict[str, np.ndarray]] = None
    failure_code: Optional[str] = None
    wall_time: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SoftZooRolloutBatchResult:
    """Batch of rollout results."""

    results: List[SoftZooRolloutResult]
    returns: np.ndarray
    successes: np.ndarray
    wall_time_total: float = 0.0
