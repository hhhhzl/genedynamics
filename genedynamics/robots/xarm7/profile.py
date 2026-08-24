"""Canonical xArm7 profile backed by the vendored MuJoCo Menagerie model."""

from pathlib import Path

from genedynamics.robots.profile import (
    CARTESIAN_JACOBIAN,
    FIXED_BASE,
    SINGLE_TOOL,
    TORQUE_CONTROL,
    NamedElement,
    RobotProfile,
)


def _model_path() -> str:
    path = (
        Path(__file__).resolve().parents[3]
        / "third_party" / "mujoco_menagerie"
        / "ufactory_xarm7" / "xarm7_nohand.xml"
    )
    if not path.exists():
        raise FileNotFoundError(f"vendored xArm7 MJCF not found: {path}")
    return str(path)


def xarm7_profile() -> RobotProfile:
    return RobotProfile(
        model_id="xarm7",
        robot_type="manipulator",
        base_type=FIXED_BASE,
        model_path_resolver=_model_path,
        actuated_joints=tuple(f"joint{i}" for i in range(1, 8)),
        actuator_names=tuple(f"act{i}" for i in range(1, 8)),
        joint_groups={"arm": tuple(range(7))},
        elements={"tool_mount": NamedElement("site", "attachment_site")},
        capabilities=frozenset({
            FIXED_BASE, CARTESIAN_JACOBIAN, TORQUE_CONTROL, SINGLE_TOOL,
        }),
        controller_defaults={
            "home_qpos": (0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0),
            "torque_limits": (50.0, 50.0, 30.0, 30.0, 30.0, 20.0, 20.0),
            "replace_actuators_with_motors": True,
            "strip_mesh_geoms": True,
        },
    )
