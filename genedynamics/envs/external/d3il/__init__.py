"""
D3IL integration layer (external environment).

This package contains *only* adapters and bootstrapping helpers so that the
third-party D3IL code can live in `third_party/environments/d3il/` (or a pinned
git submodule), without polluting the core genedynamics package.
"""

from .bootstrap import ensure_d3il_on_path

# Ensure path and pinocchio compat as soon as d3il is used, before any third_party code loads
ensure_d3il_on_path()
from .avoiding_env import D3ILAvoidingEnv, D3ILAvoidingConfig
from .avoiding_plan_env_9d import AvoidingPlanEnv9D, AvoidingPlanSpec9D
from .avoiding_env_7d_vel import D3ILAvoiding7dVelEnv, D3ILAvoiding7dVelConfig
from .task_env import D3ILTaskEnv
from .specs import D3ILTaskSpec, D3ILTaskConfig, D3ILAvoidingSpec, D3ILAvoidingSpecConfig

__all__ = [
    "ensure_d3il_on_path",
    "D3ILTaskEnv",
    "D3ILTaskSpec",
    "D3ILTaskConfig",
    "D3ILAvoidingSpec",
    "D3ILAvoidingSpecConfig",
    "D3ILAvoidingEnv",
    "D3ILAvoidingConfig",
    "AvoidingPlanEnv9D",
    "AvoidingPlanSpec9D",
    "D3ILAvoiding7dVelEnv",
    "D3ILAvoiding7dVelConfig",
]


