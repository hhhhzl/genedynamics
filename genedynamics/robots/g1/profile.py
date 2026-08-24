"""Canonical task-independent G1 description used by MJX tasks."""

from pathlib import Path

from genedynamics.robots.g1.assets import g1_mjcf_path
from genedynamics.robots.g1.spec import G1_ACTUATED_JOINTS
from genedynamics.robots.profile import (
    BIPED,
    FLOATING_BASE,
    TORQUE_CONTROL,
    TWO_FEET,
    TWO_HANDS,
    WHOLE_BODY_CONTROL,
    NamedElement,
    RobotProfile,
)


def _box_scene() -> str:
    return str(
        Path(__file__).resolve().parents[2]
        / "envs" / "assets" / "unitree_g1" / "mjx_scene_g1_box_push.xml"
    )


def g1_profile() -> RobotProfile:
    return RobotProfile(
        model_id="g1",
        robot_type="humanoid",
        base_type=FLOATING_BASE,
        model_path_resolver=lambda: g1_mjcf_path(prefer_mjx=True),
        actuated_joints=G1_ACTUATED_JOINTS,
        joint_groups={
            "planner": tuple(range(15)),
            "left_arm": tuple(range(15, 22)),
            "right_arm": tuple(range(22, 29)),
            "left_sagittal_leg": (0, 3, 4),
            "right_sagittal_leg": (6, 9, 10),
            "hip_roll": (1, 7),
        },
        elements={
            "pelvis": NamedElement("body", "pelvis"),
            "torso": NamedElement("body", "torso_link"),
            "left_foot": NamedElement("site", "left_foot"),
            "right_foot": NamedElement("site", "right_foot"),
            "left_hand": NamedElement("body", "left_wrist_yaw_link"),
            "right_hand": NamedElement("body", "right_wrist_yaw_link"),
            "left_hand_contactor": NamedElement("geom", "left_hand_collision"),
            "right_hand_contactor": NamedElement("geom", "right_hand_collision"),
        },
        capabilities=frozenset({
            FLOATING_BASE, BIPED, TWO_FEET, TWO_HANDS,
            TORQUE_CONTROL, WHOLE_BODY_CONTROL,
        }),
        scene_resolvers={"box_push": _box_scene},
        controller_defaults={
            "hand_forward_extent": 0.076,
            "hand_push_axis": (0.0, 0.0, 1.0),
            "home_qpos": (
                0.0, 0.0, 0.786, 1.0, 0.0, 0.0, 0.0,
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                0.0, 0.0, 0.0,
                0.2, 0.2, 0.0, 1.28, 0.0, 0.0, 0.0,
                0.2, -0.2, 0.0, 1.28, 0.0, 0.0, 0.0,
            ),
            "torque_limits": tuple(
                [88.0, 139.0, 88.0, 139.0, 50.0, 50.0] * 2
                + [88.0, 50.0, 50.0]
                + [25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0] * 2
            ),
            "kp": tuple(
                [75.0, 75.0, 75.0, 75.0, 20.0, 20.0] * 2
                + [75.0, 75.0, 75.0]
                + [75.0, 75.0, 75.0, 75.0, 20.0, 20.0, 20.0] * 2
            ),
            "kd": (2.0,) * 29,
        },
    )
