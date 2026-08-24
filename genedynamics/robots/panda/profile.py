"""Canonical task-independent Panda description."""

from pathlib import Path

from genedynamics.robots.profile import (
    CARTESIAN_JACOBIAN,
    FIXED_BASE,
    SINGLE_TOOL,
    TORQUE_CONTROL,
    NamedElement,
    RobotProfile,
)


def _panda_model_path() -> str:
    return str(
        Path(__file__).resolve().parents[2]
        / "envs" / "assets" / "franka_panda" / "panda_arm.xml"
    )


def panda_profile() -> RobotProfile:
    return RobotProfile(
        model_id="panda",
        robot_type="manipulator",
        base_type=FIXED_BASE,
        model_path_resolver=_panda_model_path,
        actuated_joints=tuple(f"joint{i}" for i in range(1, 8)),
        actuator_names=tuple(f"m{i}" for i in range(1, 8)),
        joint_groups={"arm": tuple(range(7))},
        elements={"tool_mount": NamedElement("site", "ee")},
        capabilities=frozenset({
            FIXED_BASE, CARTESIAN_JACOBIAN, TORQUE_CONTROL, SINGLE_TOOL,
        }),
        controller_defaults={
            "home_qpos": (0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.78),
            "torque_limits": (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
        },
    )
