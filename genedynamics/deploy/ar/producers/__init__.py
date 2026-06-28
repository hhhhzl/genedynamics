"""Producers: adapters that write domain data into a generic ``WorldState``.

Each producer is small and independent (corridor scene, tracker poses, ROS2
topics). Multiple producers write into ONE ``WorldState`` keyed by entity id —
which is how multi-robot / multi-arm scenes compose without new architecture.
"""

from __future__ import annotations

from genedynamics.deploy.ar.producers.corridor import corridor_entities, populate_corridor
from genedynamics.deploy.ar.producers.ros2 import (
    Ros2Bridge,
    marker_to_entity,
    markerarray_to_entities,
    ros_pose_to_pose,
    ros_transform_to_pose,
)
from genedynamics.deploy.ar.producers.tracker_pose import (
    headset_entities,
    populate_headset,
    populate_tracker,
    robot_entities,
)

__all__ = [
    "corridor_entities", "populate_corridor",
    "robot_entities", "populate_tracker",
    "headset_entities", "populate_headset",
    "Ros2Bridge", "marker_to_entity", "markerarray_to_entities",
    "ros_pose_to_pose", "ros_transform_to_pose",
]
