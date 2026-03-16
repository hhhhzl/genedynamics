"""
Task registry for SoftZoo tasks.

Central registry for task specs. Enables extension without modifying
core adapter code. Tasks are registered by task_id.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type

from .schemas import (
    SoftZooTaskSpec,
    SoftZooEnvType,
    SoftZooObjective,
    SoftZooMorphologySpec,
    SoftZooControllerSpec,
    SoftZooModeSpec,
    SoftZooFidelitySpec,
    SoftZooRobotSpec,
)


# Default mode specs for friction regimes
DEFAULT_MODES = (
    SoftZooModeSpec(mode_id=0, name="low_friction", friction=0.2, log_prior=0.0),
    SoftZooModeSpec(mode_id=1, name="medium_friction", friction=0.5, log_prior=0.0),
    SoftZooModeSpec(mode_id=2, name="high_friction", friction=0.8, log_prior=0.0),
    SoftZooModeSpec(mode_id=3, name="very_high_friction", friction=1.1, log_prior=0.0),
)

# Default fidelity levels
DEFAULT_FIDELITY_LEVELS = (
    SoftZooFidelitySpec(level=0, max_substeps=500, max_substeps_local=10, n_frames=100),
    SoftZooFidelitySpec(level=1, max_substeps=1500, max_substeps_local=15, n_frames=150),
    SoftZooFidelitySpec(level=2, max_substeps=3500, max_substeps_local=20, n_frames=200),
)


_REGISTRY: Dict[str, SoftZooTaskSpec] = {}


def register_task(
    task_id: str,
    spec: Optional[SoftZooTaskSpec] = None,
    *,
    factory: Optional[Callable[[], SoftZooTaskSpec]] = None,
) -> None:
    """
    Register a task spec.

    Args:
        task_id: Unique task identifier
        spec: Task spec (if provided directly)
        factory: Callable that returns spec (used for lazy init)
    """
    if spec is not None:
        _REGISTRY[task_id] = spec
    elif factory is not None:
        _REGISTRY[task_id] = factory()
    else:
        raise ValueError("Either spec or factory must be provided")


def get_task_spec(task_id: str) -> SoftZooTaskSpec:
    """Get task spec by id. Raises KeyError if not found."""
    if task_id not in _REGISTRY:
        raise KeyError(f"Task '{task_id}' not registered. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[task_id]


def list_tasks() -> List[str]:
    """List registered task ids."""
    return list(_REGISTRY.keys())


def _register_defaults() -> None:
    """Register default tasks."""
    # Crawling on flat ground
    register_task(
        "crawling_ground",
        SoftZooTaskSpec(
            task_id="crawling_ground",
            env_type=SoftZooEnvType.LAND,
            env_config_file="ground.yaml",
            objective=SoftZooObjective.MOVE_FORWARD,
            robot=SoftZooRobotSpec(
                morphology=SoftZooMorphologySpec(pcd_name="Caterpillar"),
                controller=SoftZooControllerSpec(n_actuators=10),
            ),
            modes=DEFAULT_MODES,
            fidelity_levels=DEFAULT_FIDELITY_LEVELS,
            max_steps=200,
        ),
    )

    # Crawling on desert
    register_task(
        "crawling_desert",
        SoftZooTaskSpec(
            task_id="crawling_desert",
            env_type=SoftZooEnvType.LAND,
            env_config_file="desert.yaml",
            objective=SoftZooObjective.MOVE_FORWARD,
            robot=SoftZooRobotSpec(
                morphology=SoftZooMorphologySpec(pcd_name="Caterpillar"),
                controller=SoftZooControllerSpec(n_actuators=10),
            ),
            modes=DEFAULT_MODES,
            fidelity_levels=DEFAULT_FIDELITY_LEVELS,
            max_steps=200,
        ),
    )

    # Obstacle crossing (use ground with obstacles if available)
    register_task(
        "obstacle_crossing",
        SoftZooTaskSpec(
            task_id="obstacle_crossing",
            env_type=SoftZooEnvType.LAND,
            env_config_file="ground.yaml",
            objective=SoftZooObjective.MOVE_FORWARD,
            robot=SoftZooRobotSpec(
                morphology=SoftZooMorphologySpec(pcd_name="Caterpillar"),
                controller=SoftZooControllerSpec(n_actuators=10),
            ),
            modes=DEFAULT_MODES,
            fidelity_levels=DEFAULT_FIDELITY_LEVELS,
            max_steps=200,
        ),
    )


# Auto-register defaults on import
_register_defaults()


class TaskRegistry:
    """
    Fluent API for task registration.

    Usage:
        TaskRegistry().register("my_task", my_spec).freeze()
    """

    def __init__(self) -> None:
        self._pending: Dict[str, SoftZooTaskSpec] = {}

    def register(self, task_id: str, spec: SoftZooTaskSpec) -> "TaskRegistry":
        """Add task to pending registration."""
        self._pending[task_id] = spec
        return self

    def freeze(self) -> None:
        """Apply pending registrations."""
        for tid, spec in self._pending.items():
            register_task(tid, spec)
        self._pending.clear()
