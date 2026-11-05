"""Versioned G1 actuator profiles; SuperMimic gains are explicit."""

from .supermimic import (
    DAMPING_RATIO,
    NATURAL_FREQ,
    action_scale,
    default_joint_pos_list,
    motor_of,
)

__all__ = [
    "DAMPING_RATIO",
    "NATURAL_FREQ",
    "action_scale",
    "default_joint_pos_list",
    "motor_of",
]
