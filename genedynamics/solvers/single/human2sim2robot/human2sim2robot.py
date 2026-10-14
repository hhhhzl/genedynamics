"""Human2Sim2Robot native PyTorch policy solver.

The learner and video/digital-twin pipeline remain the pinned native training
process. This frontend deploys its complete actor checkpoint, including input
normalization and recurrent state; it does not train during action selection.
"""

from __future__ import annotations

from typing import Any

from .backends.human2sim2robot_torch import Human2Sim2RobotBackendTorch


class Human2Sim2RobotSolver:
    """Host-loop policy solver for the public native task observation interface."""

    def __init__(self, **kwargs: Any) -> None:
        self._backend_impl = Human2Sim2RobotBackendTorch(**kwargs)

    @property
    def model(self):
        return self._backend_impl.model

    @property
    def states(self):
        return self._backend_impl.states

    @property
    def checkpoint_metadata(self):
        return dict(self._backend_impl.checkpoint_metadata)

    def act(self, observation: Any, **kwargs: Any):
        return self._backend_impl.act(observation, **kwargs)

    def infer(self, observation: Any, **kwargs: Any):
        return self._backend_impl.infer(observation, **kwargs)

    def reset(self, batch_size: int | None = None, *, dones: Any = None) -> None:
        self._backend_impl.reset(batch_size, dones=dones)

    def restore(self, checkpoint: Any, **kwargs: Any) -> None:
        self._backend_impl.restore(checkpoint, **kwargs)


__all__ = ["Human2Sim2RobotSolver"]
