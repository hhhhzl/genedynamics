"""
Quadruped gait adapter placeholder.
"""

from __future__ import annotations

from genedynamics.deploy.followers.common.traversal_intent import TraversalIntent
from genedynamics.deploy.followers.quadruped.task_spec import QuadrupedTaskSpec


class QuadrupedGaitAdapter:
    def adapt(self, intent: TraversalIntent) -> QuadrupedTaskSpec:
        _ = intent
        raise NotImplementedError("Quadruped corridor gait adapter is not implemented yet.")
