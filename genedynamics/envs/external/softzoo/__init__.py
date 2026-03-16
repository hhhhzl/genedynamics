"""
SoftZoo external integration for soft robot co-design.

Provides bootstrap, schemas, config, task registry, and adapters for
industrial-grade SoftZoo integration with genedynamics.
"""

from __future__ import annotations

from .bootstrap import (
    ensure_softzoo_on_path,
    get_softzoo_paths,
    SoftZooPaths,
    validate_softzoo_environment,
)
from .config import SoftZooRuntimeConfig, SoftZooEnvConfig
from .schemas import (
    SoftZooTaskSpec,
    SoftZooRobotSpec,
    SoftZooMorphologySpec,
    SoftZooControllerSpec,
    SoftZooModeSpec,
    SoftZooFidelitySpec,
    SoftZooRolloutRequest,
    SoftZooRolloutResult,
)
from .task_registry import (
    TaskRegistry,
    register_task,
    get_task_spec,
    list_tasks,
)
from .adapters import (
    make_softzoo_env,
    encode_morphology,
    encode_controller,
    decode_rollout,
)

__all__ = [
    # Bootstrap
    "ensure_softzoo_on_path",
    "get_softzoo_paths",
    "SoftZooPaths",
    "validate_softzoo_environment",
    # Config
    "SoftZooRuntimeConfig",
    "SoftZooEnvConfig",
    # Schemas
    "SoftZooTaskSpec",
    "SoftZooRobotSpec",
    "SoftZooMorphologySpec",
    "SoftZooControllerSpec",
    "SoftZooModeSpec",
    "SoftZooFidelitySpec",
    "SoftZooRolloutRequest",
    "SoftZooRolloutResult",
    # Task registry
    "TaskRegistry",
    "register_task",
    "get_task_spec",
    "list_tasks",
    # Adapters
    "make_softzoo_env",
    "encode_morphology",
    "encode_controller",
    "decode_rollout",
]
