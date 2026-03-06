"""
Deploy pipeline: unified config, modes, profiles, and CLI.

Usage:
    genedynamics-deploy --config configs/quadruped/flat/mbd_deploy.yaml
    genedynamics-deploy --robot quadruped --model go2 --mode sim --planner mbd
"""

from genedynamics.deploy.config import DeployConfig, SimConfig, RealConfig, ReplayConfig, StartConfig
from genedynamics.deploy.pipeline import run_pipeline
from genedynamics.deploy.profiles import get_profile_registry
from genedynamics.deploy.modes import get_mode
from genedynamics.deploy.task_config import (
    TaskConfig,
    ObstacleConfig,
    TerrainConfig,
    PerturbationConfig,
    VelocityTaskConfig,
    SequenceTaskConfig,
)

__all__ = [
    "DeployConfig",
    "SimConfig",
    "RealConfig",
    "ReplayConfig",
    "StartConfig",
    "TaskConfig",
    "ObstacleConfig",
    "TerrainConfig",
    "PerturbationConfig",
    "VelocityTaskConfig",
    "SequenceTaskConfig",
    "run_pipeline",
    "get_profile_registry",
    "get_mode",
]
