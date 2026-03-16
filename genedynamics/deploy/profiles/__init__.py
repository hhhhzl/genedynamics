"""
Robot profiles for deploy pipeline.

Each profile provides: make_env, make_energy, make_planner, infer_spec.
Profiles are registered by (robot_type, model_id) and drive the pipeline.
"""

from genedynamics.deploy.profiles.base import RobotProfile, get_profile_registry
from genedynamics.deploy.profiles.quadruped import create_quadruped_profiles
from genedynamics.deploy.profiles.humanoid import create_humanoid_profiles
from genedynamics.deploy.profiles.uav3d import create_uav3d_profiles

__all__ = [
    "RobotProfile",
    "get_profile_registry",
    "create_quadruped_profiles",
    "create_humanoid_profiles",
    "create_uav3d_profiles",
]

# Register built-in profiles on import
def _register_builtins() -> None:
    reg = get_profile_registry()
    create_quadruped_profiles(reg)
    create_humanoid_profiles(reg)
    create_uav3d_profiles(reg)


_register_builtins()
