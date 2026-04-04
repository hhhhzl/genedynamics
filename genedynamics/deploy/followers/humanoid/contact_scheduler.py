"""
Humanoid contact manager for corridor following.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from genedynamics.deploy.followers.common.plan_schema import CorridorPlanFrame
from genedynamics.deploy.followers.common.traversal_intent import TraversalIntent
from genedynamics.deploy.followers.humanoid.corridor_envelope import (
    CorridorEnvelopeConfig,
    CorridorEnvelopeHeuristics,
)
from genedynamics.deploy.followers.humanoid.task_spec import (
    ContactObservations,
    ContactPhase,
    PhaseState,
)


@dataclass
class HumanoidContactSchedulerConfig:
    double_support_time: float = 0.12
    double_support_time_through_gap: float = 0.22
    swing_time: float = 0.42
    swing_time_through_gap: float = 0.58
    min_motion_score: float = 0.06
    progress_gate: float = 0.05
    yaw_rate_weight: float = 0.10
    initial_swing_foot: str = "left"
    min_touchdown_alpha: float = 0.62
    touchdown_height_threshold: float = 0.045
    touchdown_vz_threshold: float = 0.05
    touchdown_load_threshold: float = 45.0
    touchdown_load_threshold_through_gap: float = 30.0
    liftoff_height_threshold: float = 0.03
    liftoff_unload_threshold: float = 12.0
    max_liftoff_wait: float = 0.18
    support_loss_timeout: float = 0.08
    double_support_load_threshold: float = 35.0
    swing_support_load_threshold: float = 70.0
    startup_double_support_steps: int = 2
    startup_double_support_scale: float = 2.5
    startup_swing_support_load_scale: float = 1.25
    envelope: CorridorEnvelopeConfig = field(default_factory=CorridorEnvelopeConfig)


class HumanoidContactScheduler:
    """
    Contact manager with basic touchdown and liftoff logic.

    The scheduler still follows an alternating nominal gait, but transitions
    are gated by observed foot contacts instead of pure elapsed time.
    """

    def __init__(self, cfg: Optional[HumanoidContactSchedulerConfig] = None) -> None:
        self.cfg = cfg or HumanoidContactSchedulerConfig()
        self.envelope = CorridorEnvelopeHeuristics(self.cfg.envelope)
        self.reset()

    def reset(self, initial_frame: Optional[CorridorPlanFrame] = None) -> None:
        self._phase = ContactPhase.DOUBLE_SUPPORT
        self._elapsed = 0.0
        self._step_index = 0
        self._next_swing_foot = "left" if str(self.cfg.initial_swing_foot).lower().startswith("l") else "right"
        self._prev_xy = None if initial_frame is None else np.asarray(initial_frame.xy, dtype=np.float64)
        self._liftoff_confirmed = False
        self._support_loss_elapsed = 0.0

    def advance(
        self,
        intent: TraversalIntent,
        dt: float,
        observations: ContactObservations,
    ) -> PhaseState:
        dt = float(max(dt, 1e-6))
        xy = np.asarray(intent.planar_position, dtype=np.float64)
        progress = 0.0 if self._prev_xy is None else float(np.linalg.norm(xy - self._prev_xy))
        motion_score = float(np.linalg.norm(intent.planar_velocity)) + float(self.cfg.yaw_rate_weight) * abs(intent.yaw_rate)
        motion_score = max(motion_score, progress / dt)
        envelope_state = self.envelope.evaluate(intent)

        transitioned = False
        touchdown_detected = False
        liftoff_detected = False
        support_ok = self._support_contact_ok(observations, self._double_support_load_threshold())
        self._elapsed += dt

        if self._phase == ContactPhase.DOUBLE_SUPPORT:
            if (
                motion_score >= float(self.cfg.min_motion_score)
                and self._elapsed >= self._double_support_duration(envelope_state.through_gap_mode)
                and support_ok
            ):
                self._phase = ContactPhase.LEFT_SWING if self._next_swing_foot == "left" else ContactPhase.RIGHT_SWING
                self._elapsed = 0.0
                self._liftoff_confirmed = False
                self._support_loss_elapsed = 0.0
                transitioned = True
        else:
            swing_foot = "left" if self._phase == ContactPhase.LEFT_SWING else "right"
            support_foot = "right" if swing_foot == "left" else "left"
            swing_obs = observations.left if swing_foot == "left" else observations.right
            support_obs = observations.right if swing_foot == "left" else observations.left

            if (
                not swing_obs.in_contact
                or float(swing_obs.position_world[2]) >= float(self.cfg.liftoff_height_threshold)
                or float(swing_obs.support_load) <= float(self.cfg.liftoff_unload_threshold)
            ):
                self._liftoff_confirmed = True
                liftoff_detected = True
            elif self._elapsed >= float(self.cfg.max_liftoff_wait):
                self._liftoff_confirmed = True

            if support_obs.in_contact and float(support_obs.support_load) >= self._swing_support_load_threshold():
                self._support_loss_elapsed = 0.0
            else:
                self._support_loss_elapsed += dt

            alpha = self._phase_alpha(self._swing_duration(envelope_state.through_gap_mode))
            touchdown_detected = bool(
                alpha >= float(self.cfg.min_touchdown_alpha)
                and swing_obs.in_contact
                and float(swing_obs.position_world[2]) <= float(self.cfg.touchdown_height_threshold)
                and float(swing_obs.velocity_world[2]) <= float(self.cfg.touchdown_vz_threshold)
                and float(swing_obs.support_load) >= self._touchdown_load_threshold(envelope_state.through_gap_mode)
            )
            timeout = self._elapsed >= self._swing_duration(envelope_state.through_gap_mode)
            support_lost = self._support_loss_elapsed >= float(self.cfg.support_loss_timeout)
            progress_reached = (
                progress >= float(self.cfg.progress_gate)
                and self._liftoff_confirmed
                and float(swing_obs.support_load) >= self._touchdown_load_threshold(envelope_state.through_gap_mode)
            )

            if touchdown_detected or timeout or support_lost or progress_reached:
                self._phase = ContactPhase.DOUBLE_SUPPORT
                self._elapsed = 0.0
                self._step_index += 1
                self._next_swing_foot = support_foot
                self._liftoff_confirmed = False
                self._support_loss_elapsed = 0.0
                transitioned = True

        duration = self._current_phase_duration(envelope_state.through_gap_mode)
        alpha = self._phase_alpha(duration)
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
                "through_gap_mode": envelope_state.through_gap_mode,
                "narrowness": envelope_state.narrowness,
                "gap_severity": envelope_state.gap_severity,
                "liftoff_confirmed": self._liftoff_confirmed,
                "liftoff_detected": liftoff_detected,
                "touchdown_detected": touchdown_detected,
                "left_contact": bool(observations.left.in_contact),
                "right_contact": bool(observations.right.in_contact),
                "left_support_load": float(observations.left.support_load),
                "right_support_load": float(observations.right.support_load),
                "support_loss_elapsed": float(self._support_loss_elapsed),
            },
        )

    def _current_phase_duration(self, through_gap_mode: bool) -> float:
        if self._phase == ContactPhase.DOUBLE_SUPPORT:
            return self._double_support_duration(through_gap_mode)
        return self._swing_duration(through_gap_mode)

    def _double_support_duration(self, through_gap_mode: bool) -> float:
        duration = float(
            self.cfg.double_support_time_through_gap
            if through_gap_mode
            else self.cfg.double_support_time
        )
        if int(self._step_index) < int(self.cfg.startup_double_support_steps):
            duration *= float(self.cfg.startup_double_support_scale)
        return duration

    def _swing_duration(self, through_gap_mode: bool) -> float:
        return float(
            self.cfg.swing_time_through_gap
            if through_gap_mode
            else self.cfg.swing_time
        )

    def _phase_alpha(self, duration: float) -> float:
        return 1.0 if duration <= 1e-6 else float(np.clip(self._elapsed / duration, 0.0, 1.0))

    def _touchdown_load_threshold(self, through_gap_mode: bool) -> float:
        return float(
            self.cfg.touchdown_load_threshold_through_gap
            if through_gap_mode
            else self.cfg.touchdown_load_threshold
        )

    def _double_support_load_threshold(self) -> float:
        return float(self.cfg.double_support_load_threshold) * (
            float(self.cfg.startup_swing_support_load_scale)
            if int(self._step_index) < int(self.cfg.startup_double_support_steps)
            else 1.0
        )

    def _swing_support_load_threshold(self) -> float:
        return float(self.cfg.swing_support_load_threshold) * (
            float(self.cfg.startup_swing_support_load_scale)
            if int(self._step_index) < int(self.cfg.startup_double_support_steps)
            else 1.0
        )

    @staticmethod
    def _phase_to_contacts(phase: ContactPhase) -> Tuple[Tuple[str, ...], Optional[str]]:
        if phase == ContactPhase.LEFT_SWING:
            return ("right",), "left"
        if phase == ContactPhase.RIGHT_SWING:
            return ("left",), "right"
        return ("left", "right"), None

    @staticmethod
    def _support_contact_ok(observations: ContactObservations, load_threshold: float) -> bool:
        return bool(
            observations.left.in_contact
            and observations.right.in_contact
            and float(observations.left.support_load) >= float(load_threshold)
            and float(observations.right.support_load) >= float(load_threshold)
        )


# Compatibility aliases for the existing humanoid_corridor package.
HumanoidPhaseSchedulerConfig = HumanoidContactSchedulerConfig
QuasiStaticHumanoidPhaseScheduler = HumanoidContactScheduler
