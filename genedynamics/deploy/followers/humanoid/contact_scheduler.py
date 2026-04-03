"""
Humanoid contact scheduler for corridor following.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from genedynamics.deploy.followers.common.plan_schema import CorridorPlanFrame
from genedynamics.deploy.followers.humanoid.task_spec import (
    ContactPhase,
    PhaseState,
)


@dataclass
class HumanoidContactSchedulerConfig:
    double_support_time: float = 0.12
    swing_time: float = 0.32
    min_motion_score: float = 0.06
    progress_gate: float = 0.05
    yaw_rate_weight: float = 0.10
    initial_swing_foot: str = "left"


class HumanoidContactScheduler:
    """
    Minimal alternating gait scheduler.

    This is the contact-side counterpart to the previous stage-1 phase
    scheduler. It stays intentionally small so a future touchdown-aware
    implementation can replace it without changing call sites.
    """

    def __init__(self, cfg: Optional[HumanoidContactSchedulerConfig] = None) -> None:
        self.cfg = cfg or HumanoidContactSchedulerConfig()
        self.reset()

    def reset(self, initial_frame: Optional[CorridorPlanFrame] = None) -> None:
        self._phase = ContactPhase.DOUBLE_SUPPORT
        self._elapsed = 0.0
        self._step_index = 0
        self._next_swing_foot = "left" if str(self.cfg.initial_swing_foot).lower().startswith("l") else "right"
        self._prev_xy = None if initial_frame is None else np.asarray(initial_frame.xy, dtype=np.float64)

    def advance(self, frame: CorridorPlanFrame, dt: float) -> PhaseState:
        dt = float(max(dt, 1e-6))
        xy = np.asarray(frame.xy, dtype=np.float64)
        progress = 0.0 if self._prev_xy is None else float(np.linalg.norm(xy - self._prev_xy))
        motion_score = float(np.linalg.norm([frame.v_x, frame.v_y])) + float(self.cfg.yaw_rate_weight) * abs(frame.omega)
        motion_score = max(motion_score, progress / dt)

        self._elapsed += dt
        transitioned = False
        if self._phase == ContactPhase.DOUBLE_SUPPORT:
            if motion_score >= float(self.cfg.min_motion_score) and self._elapsed >= float(self.cfg.double_support_time):
                self._phase = ContactPhase.LEFT_SWING if self._next_swing_foot == "left" else ContactPhase.RIGHT_SWING
                self._elapsed = 0.0
                transitioned = True
        else:
            if self._elapsed >= float(self.cfg.swing_time) or progress >= float(self.cfg.progress_gate):
                self._phase = ContactPhase.DOUBLE_SUPPORT
                self._elapsed = 0.0
                self._step_index += 1
                self._next_swing_foot = "right" if self._next_swing_foot == "left" else "left"
                transitioned = True

        duration = float(
            self.cfg.double_support_time
            if self._phase == ContactPhase.DOUBLE_SUPPORT
            else self.cfg.swing_time
        )
        alpha = 1.0 if duration <= 1e-6 else float(np.clip(self._elapsed / duration, 0.0, 1.0))
        support_feet, swing_foot = self._phase_to_contacts(self._phase)
        self._prev_xy = xy
        return PhaseState(
            phase=self._phase,
            alpha=alpha,
            step_index=int(self._step_index),
            swing_foot=swing_foot,
            support_feet=support_feet,
            phase_elapsed=float(self._elapsed),
            phase_duration=duration,
            metadata={
                "motion_score": motion_score,
                "progress": progress,
                "transitioned": transitioned,
                "next_swing_foot": self._next_swing_foot,
            },
        )

    @staticmethod
    def _phase_to_contacts(phase: ContactPhase) -> Tuple[Tuple[str, ...], Optional[str]]:
        if phase == ContactPhase.LEFT_SWING:
            return ("right",), "left"
        if phase == ContactPhase.RIGHT_SWING:
            return ("left",), "right"
        return ("left", "right"), None


# Compatibility aliases for the existing humanoid_corridor package.
HumanoidPhaseSchedulerConfig = HumanoidContactSchedulerConfig
QuasiStaticHumanoidPhaseScheduler = HumanoidContactScheduler
