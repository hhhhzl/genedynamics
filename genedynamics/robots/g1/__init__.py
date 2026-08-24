"""Unitree G1 robot package.

The single source of truth for the G1 humanoid:

* :mod:`assets`  — MJCF asset path resolution
* :mod:`spec`    — joint topology constants and the introspected
  :class:`G1RobotSpec` dataclass

Importing this package is idempotent and registers G1 with the global robot
registry, replacing any previously installed entry. New robots should follow
the same pattern: ``genedynamics/robots/<id>/{__init__.py, spec.py, assets.py}``.
"""

from genedynamics.robots.g1.assets import (
    G1AssetNotFoundError,
    g1_mjcf_path,
    g1_scene_path,
)
from genedynamics.robots.g1.spec import (
    G1_ACTUATED_JOINTS,
    G1_BODY_NAMES,
    G1_FOOT_BODY_NAMES,
    G1_FOOT_SITE_NAMES,
    G1_IMU_SITE_NAMES,
    G1_LEFT_ARM_JOINTS,
    G1_LEFT_LEG_JOINTS,
    G1_RIGHT_ARM_JOINTS,
    G1_RIGHT_LEG_JOINTS,
    G1_WAIST_JOINTS,
    G1RobotSpec,
)
from genedynamics.robots.g1.profile import g1_profile

__all__ = [
    "G1AssetNotFoundError",
    "g1_mjcf_path",
    "g1_scene_path",
    "G1RobotSpec",
    "G1_ACTUATED_JOINTS",
    "G1_BODY_NAMES",
    "G1_FOOT_BODY_NAMES",
    "G1_FOOT_SITE_NAMES",
    "G1_IMU_SITE_NAMES",
    "G1_LEFT_ARM_JOINTS",
    "G1_LEFT_LEG_JOINTS",
    "G1_RIGHT_ARM_JOINTS",
    "G1_RIGHT_LEG_JOINTS",
    "G1_WAIST_JOINTS",
    "g1_profile",
]


def _register_with_global_registry() -> None:
    """Register G1 with :mod:`genedynamics.robots.registry`.

    Idempotent: if an entry for ``("humanoid", "g1")`` is already present
    (e.g. installed by ``_register_builtins``) it is overwritten with one
    whose ``model_path_resolver`` points at this package's resolver.
    """
    try:
        from genedynamics.robots.registry import get_robot_registry
        from genedynamics.tasks.humanoid.spec import HumanoidTaskSpec
    except Exception:
        # Robot registry / task spec not importable in some minimal environments;
        # the package itself remains usable.
        return

    get_robot_registry().register(
        robot_type="humanoid",
        model_id="g1",
        env_factory_name="humanoid_g1_physics",
        nq=36,
        nv=35,
        act_dim=29,
        model_path_resolver=g1_scene_path,
        spec_class=HumanoidTaskSpec,
        profile_factory=g1_profile,
        description="Unitree G1 (mujoco_menagerie)",
    )


_register_with_global_registry()
