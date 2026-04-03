"""
Compatibility exports for the legacy humanoid corridor package.

New code should import from `genedynamics.deploy.followers.humanoid.models.g1_model`.
"""

from genedynamics.deploy.followers.humanoid.models.g1_model import (
    ACTUATED_JOINTS,
    FOOT_SITE_NAMES,
    G1ModelSpec,
    LEFT_ARM_JOINTS,
    LEFT_LEG_JOINTS,
    RIGHT_ARM_JOINTS,
    RIGHT_LEG_JOINTS,
    WAIST_JOINTS,
    resolve_g1_model_path,
)

__all__ = [
    "ACTUATED_JOINTS",
    "FOOT_SITE_NAMES",
    "G1ModelSpec",
    "LEFT_ARM_JOINTS",
    "LEFT_LEG_JOINTS",
    "RIGHT_ARM_JOINTS",
    "RIGHT_LEG_JOINTS",
    "WAIST_JOINTS",
    "resolve_g1_model_path",
]
