"""Canonical task-independent H1 description."""

from pathlib import Path

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


_JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    "torso",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
)


def _asset(name: str) -> str:
    return str(
        Path(__file__).resolve().parents[2] / "envs" / "assets" / "unitree_h1" / name
    )


def h1_profile() -> RobotProfile:
    return RobotProfile(
        model_id="h1",
        robot_type="humanoid",
        base_type=FLOATING_BASE,
        model_path_resolver=lambda: _asset("mjx_h1_calf_hand_body.xml"),
        actuated_joints=_JOINTS,
        joint_groups={
            "planner": tuple(range(11)),
            "left_arm": (11, 12, 13, 14),
            "right_arm": (15, 16, 17, 18),
            "left_sagittal_leg": (2, 3, 4),
            "right_sagittal_leg": (7, 8, 9),
            "hip_roll": (1, 6),
        },
        elements={
            "pelvis": NamedElement("body", "pelvis"),
            "torso": NamedElement("body", "torso_link"),
            "left_foot": NamedElement("site", "left_foot"),
            "right_foot": NamedElement("site", "right_foot"),
            "left_hand": NamedElement("body", "left_elbow_link"),
            "right_hand": NamedElement("body", "right_elbow_link"),
            "left_hand_contactor": NamedElement("geom", "left_hand_collision"),
            "right_hand_contactor": NamedElement("geom", "right_hand_collision"),
        },
        capabilities=frozenset({
            FLOATING_BASE, BIPED, TWO_FEET, TWO_HANDS,
            TORQUE_CONTROL, WHOLE_BODY_CONTROL,
        }),
        scene_resolvers={"box_push": lambda: _asset("mjx_scene_h1_box_push.xml")},
        controller_defaults={
            "hand_forward_extent": 0.033,
            "hand_push_axis": (1.0, 0.0, 0.0),
            "home_qpos": (
                0.0, 0.0, 0.98, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, -0.4, 0.8, -0.4,
                0.0, 0.0, -0.4, 0.8, -0.4,
                0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
            ),
            "kp": (
                200.0, 200.0, 200.0, 200.0, 60.0,
                200.0, 200.0, 200.0, 200.0, 60.0,
                200.0,
                60.0, 60.0, 60.0, 60.0,
                60.0, 60.0, 60.0, 60.0,
            ),
            "kd": (
                5.0, 5.0, 5.0, 5.0, 1.5,
                5.0, 5.0, 5.0, 5.0, 1.5,
                5.0,
                1.5, 1.5, 1.5, 1.5,
                1.5, 1.5, 1.5, 1.5,
            ),
        },
    )
