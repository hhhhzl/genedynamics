"""Compose the box-push task onto any compatible humanoid robot MJCF."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from genedynamics.envs.composition import MjcfSceneComposer


def compose_box_push_model(profile):
    composer = MjcfSceneComposer(profile)
    defaults = profile.controller_defaults
    torque_limits = defaults.get("torque_limits")
    if torque_limits is not None:
        composer.replace_actuators_with_motors(torque_limits)

    world = composer.worldbody
    floor_attrs = {
        "name": "floor", "type": "plane", "size": "0 0 0.05",
        "rgba": "0.32 0.42 0.52 1",
    }
    if profile.model_id == "h1":
        floor_attrs.update({"contype": "2", "conaffinity": "1"})
    ET.SubElement(world, "geom", floor_attrs)
    ET.SubElement(world, "site", {
        "name": "goal_line", "pos": "0.85 0 0.02", "size": "0.02 1 0.02",
        "type": "box", "rgba": "0.1 0.9 0.2 0.4",
    })
    for name, y in (("unjam_wall_left", 0.64), ("unjam_wall_right", -0.64)):
        ET.SubElement(world, "geom", {
            "name": name, "type": "box", "pos": f"1.20 {y} 0.60",
            "size": "0.60 0.05 0.60", "contype": "1" if profile.model_id == "h1" else "0",
            "conaffinity": "4" if profile.model_id == "h1" else "0",
            "rgba": "0.35 0.38 0.45 0.75",
        })

    box = ET.SubElement(world, "body", {"name": "box_body", "pos": "0 0 0.55"})
    box_contact = {"contype": "4", "conaffinity": "1"} if profile.model_id == "h1" else {
        "contype": "0", "conaffinity": "0"
    }
    ET.SubElement(box, "geom", {
        "name": "static_box", "type": "box", "size": "0.55 0.55 0.55",
        "friction": "0.6", "rgba": "0.87 0.72 0.53 0.6", **box_contact,
    })
    ET.SubElement(box, "inertial", {
        "mass": "30", "pos": "0 0 0", "diaginertia": "0.1 0.1 0.1",
    })
    for name, joint_type, axis, friction in (
        ("box_x", "slide", "1 0 0", "12"),
        ("box_y", "slide", "0 1 0", "20"),
        ("box_yaw", "hinge", "0 0 1", "8"),
    ):
        ET.SubElement(box, "joint", {
            "name": name, "type": joint_type, "axis": axis, "frictionloss": friction,
        })

    contact = composer.contact
    if profile.model_id == "g1":
        for side, prefix in (("l", "left"), ("r", "right")):
            for index in (1, 2, 3):
                ET.SubElement(contact, "pair", {
                    "name": f"{side}f{index}_floor",
                    "geom1": f"{prefix}_foot{index}_collision",
                    "geom2": "floor", "solref": "0.012 1",
                    "friction": "1 1", "condim": "3",
                })
    left_geom = profile.elements["left_hand_contactor"].name
    right_geom = profile.elements["right_hand_contactor"].name
    for name, geom in (("box_lhand", left_geom), ("box_rhand", right_geom)):
        ET.SubElement(contact, "pair", {
            "name": name, "geom1": "static_box", "geom2": geom,
            "condim": "3", "friction": "0.6 0.6", "solref": "0.10 1",
            "solimp": "0.8 0.9 0.01",
        })
    for name, wall in (
        ("box_wall_left", "unjam_wall_left"),
        ("box_wall_right", "unjam_wall_right"),
    ):
        ET.SubElement(contact, "pair", {
            "name": name, "geom1": "static_box", "geom2": wall,
            "condim": "3", "friction": "0.6 0.6", "solref": "0.10 1",
            "solimp": "0.8 0.9 0.01",
        })

    robot_home = tuple(defaults["home_qpos"])
    composer.set_home_keyframe(robot_home + (0.0, 0.0, 0.0))
    return composer.compile()
