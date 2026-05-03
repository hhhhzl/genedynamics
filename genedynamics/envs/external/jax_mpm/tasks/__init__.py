"""Task wrappers for the soft-robot MPM scene.

A "task" bundles
  - a regime descriptor (terrain, friction, manipuland config)
  - a rollout call (scene → trajectories)
  - a reward / success function

Phase 2 ships two: locomotion (writeup §11 Task 1) and push (Task 2).
Carry / loco-grasp (Task 3) lives in Phase 5.
"""

from .regime import (
    RegimeSpec,
    make_train_bank,
    make_test_bank,
)
from .locomotion import LocomotionTask, evaluate_locomotion
from .push import PushTask, evaluate_push

__all__ = [
    "RegimeSpec",
    "make_train_bank",
    "make_test_bank",
    "LocomotionTask",
    "evaluate_locomotion",
    "PushTask",
    "evaluate_push",
]
