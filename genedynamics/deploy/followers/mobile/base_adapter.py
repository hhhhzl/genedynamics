"""
Mobile-base adapter placeholder.
"""

from __future__ import annotations

from genedynamics.deploy.followers.common.traversal_intent import TraversalIntent
from genedynamics.deploy.followers.mobile.task_spec import MobileBaseTaskSpec


class MobileBaseAdapter:
    def adapt(self, intent: TraversalIntent) -> MobileBaseTaskSpec:
        return MobileBaseTaskSpec(
            planar_position=intent.planar_position.copy(),
            yaw=float(intent.yaw),
            velocity=intent.planar_velocity.copy(),
            extras=dict(intent.extras),
        )
