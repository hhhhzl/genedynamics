"""Task-owned target reposition and docking execution for Go2+Z1.

The body target is public and expressed relative to the measured reset pose.
The common execution wrapper holds the manipulation request until the body has
reached and settled at that target, then delegates the docking behavior to the
selected method actor.  It never reads mechanism parameters or evaluator
state.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np


VERSION = "public-reposition-then-dock-v1"

# Two disjoint, jointly reachable target directions inside the fixed-foot
# support region.  These are task geometry, not hidden mechanism parameters.
TARGET_OFFSETS = np.asarray([
    [.018, .010, 0.],
    [.012, -.006, 0.],
], dtype=np.float32)
TARGET_BOUNDS = np.asarray([[-.030, -.030, -.15], [.030, .030, .15]], dtype=np.float32)
POSITION_TOLERANCE_M = .003
YAW_TOLERANCE_RAD = .035
BASE_SPEED_TOLERANCE_MPS = .004
YAW_RATE_TOLERANCE_RADPS = .025
TCP_SPEED_TOLERANCE_MPS = .012
STABILITY_DWELL_CONTROLS = 12
FOOT_SLIP_LIMIT_M = .012


def target_from_reset(planar_pose, mode):
    """Return the frozen task target for one measured reset pose and mode."""
    mode = int(mode)
    if mode not in (0, 1):
        raise ValueError("reposition mode must be 0 or 1")
    pose = np.asarray(planar_pose, dtype=np.float32)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError("finite planar reset pose required")
    return pose + TARGET_OFFSETS[mode]


def mobile_mating_training_config_compatible(task):
    """Check the declared reuse boundary for a T4-trained docking actor.

    The semantic suite changes only the public body-target prephase and the
    episode deadline.  Socket geometry, robot, reset, WBC execution, action
    chart, and docking goal must remain exactly equal to the T4 training task.
    """
    from dataclasses import replace
    from .mobile_mating_policy_search import MOBILE_CONFIG
    from .specs import task_spec

    if task.spec.name != "reposition_then_dock":
        return False
    expected = task_spec(
        "T4", robot="go2_z1", config_overrides=MOBILE_CONFIG,
    ).config
    actual = task.spec.config
    normalized = replace(
        actual, environment=expected.environment,
        timeout_seconds=expected.timeout_seconds,
    )
    return normalized == expected


class PublicRepositionDockExecution:
    """Gate any public docking actor behind target-directed body settling."""

    VERSION = VERSION

    def __init__(self, spec, dock_actor, *, mode):
        if (spec.name != "reposition_then_dock" or spec.alias != "T4"
                or spec.config.robot != "go2_z1" or not spec.config.mobile):
            raise ValueError("semantic Go2+Z1 reposition-then-dock task required")
        self.spec = spec
        self.dock_actor = dock_actor
        self.mode = int(mode)
        if self.mode not in (0, 1):
            raise ValueError("reposition mode must be 0 or 1")
        self.reset()

    def reset(self):
        self.stage = "reposition"
        self.stable_controls = 0
        self.dock_start_seconds = None
        self.last_body_twist = np.zeros((1, 3), dtype=np.float32)
        self.last_action = np.zeros((1, 13), dtype=np.float32)

    def get_state(self):
        return deepcopy({
            "version": self.VERSION,
            "mode": self.mode,
            "stage": self.stage,
            "stable_controls": self.stable_controls,
            "dock_start_seconds": self.dock_start_seconds,
            "last_body_twist": self.last_body_twist,
            "last_action": self.last_action,
            "dock_actor": (
                self.dock_actor.get_state()
                if hasattr(self.dock_actor, "get_state") else None
            ),
        })

    @staticmethod
    def _settled(public):
        actual = np.asarray(public["body_planar_pose"], dtype=float)
        target = np.asarray(public["body_target"], dtype=float)
        twist = np.asarray(public["body_twist"], dtype=float)
        tcp_twist = np.asarray(public["tcp_twist"], dtype=float)
        feedback = np.asarray(public["controller_feedback"], dtype=float)
        return (
            np.linalg.norm(target[:, :2] - actual[:, :2], axis=1)
            <= POSITION_TOLERANCE_M
        ) & (
            np.abs(target[:, 2] - actual[:, 2]) <= YAW_TOLERANCE_RAD
        ) & (
            np.linalg.norm(twist[:, :2], axis=1) <= BASE_SPEED_TOLERANCE_MPS
        ) & (
            np.abs(twist[:, 5]) <= YAW_RATE_TOLERANCE_RADPS
        ) & (
            np.linalg.norm(tcp_twist[:, :3], axis=1) <= TCP_SPEED_TOLERANCE_MPS
        ) & (feedback[:, 0] <= .01) & (feedback[:, 1] <= .5)

    def body_twist(self, public):
        actual = np.asarray(public["body_planar_pose"], dtype=float)
        target = np.asarray(public["body_target"], dtype=float)
        if actual.shape != (1, 3) or target.shape != (1, 3):
            raise ValueError("reposition execution is qualified batch-one")
        if self.stage != "reposition":
            self.last_body_twist = np.zeros((1, 3), dtype=np.float32)
            return self.last_body_twist.copy()
        error = target - actual
        command = np.clip(2.0 * error, [-.010, -.010, -.025], [.010, .010, .025])
        if bool(self._settled(public)[0]):
            command[:] = 0.
        self.last_body_twist = command.astype(np.float32)
        return self.last_body_twist.copy()

    def act(self, public):
        if self.stage == "reposition":
            settled = bool(self._settled(public)[0])
            self.stable_controls = self.stable_controls + 1 if settled else 0
            if self.stable_controls >= STABILITY_DWELL_CONTROLS:
                self.stage = "dock"
                self.dock_start_seconds = float(
                    np.asarray(public["elapsed_seconds"], dtype=float).reshape(-1)[0]
                )
        if self.stage == "dock":
            dock_public = {
                key: np.asarray(value).copy() for key, value in public.items()
            }
            dock_public["elapsed_seconds"] = np.maximum(
                dock_public["elapsed_seconds"] - self.dock_start_seconds, 0.,
            )
            self.last_action = np.asarray(
                self.dock_actor.act(dock_public), dtype=np.float32,
            ).copy()
        else:
            # The WBC retains the measured TCP reference while the body moves.
            self.last_action = np.zeros((1, 13), dtype=np.float32)
        return self.last_action.copy()

    def act_with_body(self, public):
        action = self.act(public)
        return action, self.body_twist(public)


class PublicRepositionThenDockPolicy(PublicRepositionDockExecution):
    """Public source skill using the existing safe-settled docking actor."""

    def __init__(self, spec, *, mode, seed):
        from .controllers.mating_feedback import PublicLivenessMatingFeedbackPolicy

        actor = PublicDockingMethodExecution(PublicLivenessMatingFeedbackPolicy(
            spec, modes=(int(mode),), seed=int(seed),
        ))
        super().__init__(spec, actor, mode=mode)


class PublicDockingMethodExecution:
    """Project semantic observations onto the frozen docking actor schema."""

    VERSION = "public-docking-method-execution-v1"

    def __init__(self, actor):
        self.actor = actor

    def act(self, public):
        from genedynamics.solvers.single.cross_ebdm.learning.response import (
            PUBLIC_FIELDS,
        )
        return self.actor.act({key: public[key] for key in PUBLIC_FIELDS})

    def reset(self):
        return self.actor.reset()

    def get_state(self):
        return {
            "version": self.VERSION,
            "actor": (
                self.actor.get_state() if hasattr(self.actor, "get_state") else None
            ),
        }


__all__ = [
    "BASE_SPEED_TOLERANCE_MPS", "FOOT_SLIP_LIMIT_M",
    "POSITION_TOLERANCE_M", "PublicDockingMethodExecution",
    "PublicRepositionDockExecution",
    "PublicRepositionThenDockPolicy", "STABILITY_DWELL_CONTROLS",
    "TARGET_BOUNDS", "TARGET_OFFSETS", "TCP_SPEED_TOLERANCE_MPS", "VERSION",
    "YAW_RATE_TOLERANCE_RADPS", "YAW_TOLERANCE_RAD", "target_from_reset",
    "mobile_mating_training_config_compatible",
]
