"""
Footstep planning for humanoid corridor traversal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

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
class FootstepPlannerConfig:
    nominal_stance_width: float = 0.24
    nominal_foot_x: float = 0.02
    swing_height: float = 0.025
    footstep_preview_time: float = 0.35
    max_step_forward: float = 0.28
    max_step_backward: float = 0.10
    max_step_lateral: float = 0.10
    through_gap_forward_step_scale: float = 0.30
    through_gap_lateral_step_scale: float = 0.42
    through_gap_stance_scale: float = 0.58
    through_gap_preview_scale: float = 0.35
    through_gap_swing_height_scale: float = 0.85
    through_gap_center_bias_gain: float = 0.22
    through_gap_center_velocity_bias: float = 0.16
    startup_small_step_count: int = 3
    startup_forward_step_scale: float = 0.35
    startup_lateral_step_scale: float = 0.55
    startup_preview_scale: float = 0.45
    touchdown_anchor_blend: float = 0.75
    envelope: CorridorEnvelopeConfig = field(default_factory=CorridorEnvelopeConfig)


@dataclass
class FootstepPlan:
    left_position_world: np.ndarray
    left_velocity_world: np.ndarray
    right_position_world: np.ndarray
    right_velocity_world: np.ndarray
    left_anchor_world: np.ndarray
    right_anchor_world: np.ndarray
    active_swing_foot: Optional[str] = None
    swing_start_world: Optional[np.ndarray] = None
    swing_goal_world: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HumanoidFootstepPlanner:
    """
    Convert traversal intent plus contact phase into corridor-aware footstep targets.
    """

    def __init__(self, cfg: Optional[FootstepPlannerConfig] = None) -> None:
        self.cfg = cfg or FootstepPlannerConfig()
        self.envelope = CorridorEnvelopeHeuristics(self.cfg.envelope)
        self._left_anchor_world: Optional[np.ndarray] = None
        self._right_anchor_world: Optional[np.ndarray] = None
        self._left_nominal_z: float = 0.0
        self._right_nominal_z: float = 0.0
        self._swing_start_world: Optional[np.ndarray] = None
        self._swing_goal_world: Optional[np.ndarray] = None
        self._active_swing_foot: Optional[str] = None

    def reset(
        self,
        initial_intent: Optional[TraversalIntent] = None,
        *,
        left_anchor_world: Optional[np.ndarray] = None,
        right_anchor_world: Optional[np.ndarray] = None,
    ) -> None:
        if left_anchor_world is not None and right_anchor_world is not None:
            self._left_anchor_world = np.asarray(left_anchor_world, dtype=np.float64).copy()
            self._right_anchor_world = np.asarray(right_anchor_world, dtype=np.float64).copy()
            self._left_nominal_z = float(self._left_anchor_world[2])
            self._right_nominal_z = float(self._right_anchor_world[2])
        elif initial_intent is not None:
            self._left_anchor_world, self._right_anchor_world = self._default_anchor_positions(initial_intent)
        else:
            self._left_anchor_world = None
            self._right_anchor_world = None
            self._left_nominal_z = 0.0
            self._right_nominal_z = 0.0
        self._swing_start_world = None
        self._swing_goal_world = None
        self._active_swing_foot = None

    def seed_from_current_feet(self, left_world: np.ndarray, right_world: np.ndarray) -> None:
        self.reset(
            left_anchor_world=np.asarray(left_world, dtype=np.float64),
            right_anchor_world=np.asarray(right_world, dtype=np.float64),
        )

    def update(
        self,
        intent: TraversalIntent,
        phase: PhaseState,
        observations: ContactObservations,
        *,
        dt: float,
    ) -> FootstepPlan:
        _ = dt
        if self._left_anchor_world is None or self._right_anchor_world is None:
            self._left_anchor_world, self._right_anchor_world = self._default_anchor_positions(intent)

        self._left_nominal_z = float(observations.left.position_world[2])
        self._right_nominal_z = float(observations.right.position_world[2])

        if phase.metadata.get("transitioned", False):
            self._handle_transition(intent, phase, observations)

        left_pos, left_vel = self._foot_state_for_side("left", phase)
        right_pos, right_vel = self._foot_state_for_side("right", phase)
        envelope_state = self.envelope.evaluate(intent)
        return FootstepPlan(
            left_position_world=left_pos,
            left_velocity_world=left_vel,
            right_position_world=right_pos,
            right_velocity_world=right_vel,
            left_anchor_world=np.asarray(self._left_anchor_world, dtype=np.float64).copy(),
            right_anchor_world=np.asarray(self._right_anchor_world, dtype=np.float64).copy(),
            active_swing_foot=self._active_swing_foot,
            swing_start_world=None if self._swing_start_world is None else self._swing_start_world.copy(),
            swing_goal_world=None if self._swing_goal_world is None else self._swing_goal_world.copy(),
            metadata={
                "phase": phase.phase.value,
                "step_index": phase.step_index,
                "swing_foot": phase.swing_foot,
                "through_gap_mode": envelope_state.through_gap_mode,
                "narrowness": envelope_state.narrowness,
                "gap_severity": envelope_state.gap_severity,
                "effective_width": envelope_state.effective_width,
            },
        )

    def _handle_transition(
        self,
        intent: TraversalIntent,
        phase: PhaseState,
        observations: ContactObservations,
    ) -> None:
        if phase.phase == ContactPhase.DOUBLE_SUPPORT and self._active_swing_foot:
            touchdown_obs = observations.left if self._active_swing_foot == "left" else observations.right
            touchdown_world = np.asarray(touchdown_obs.position_world, dtype=np.float64)
            if self._swing_goal_world is not None:
                touchdown_world = (
                    float(self.cfg.touchdown_anchor_blend) * touchdown_world
                    + (1.0 - float(self.cfg.touchdown_anchor_blend)) * self._swing_goal_world
                )
            if self._active_swing_foot == "left":
                self._left_anchor_world = touchdown_world.copy()
            else:
                self._right_anchor_world = touchdown_world.copy()
            self._active_swing_foot = None
            self._swing_start_world = None
            self._swing_goal_world = None
            return

        if phase.phase == ContactPhase.LEFT_SWING:
            self._active_swing_foot = "left"
            self._swing_start_world = np.asarray(self._left_anchor_world, dtype=np.float64).copy()
            self._swing_goal_world = self._proposed_step_world(intent, "left", step_index=phase.step_index)
        elif phase.phase == ContactPhase.RIGHT_SWING:
            self._active_swing_foot = "right"
            self._swing_start_world = np.asarray(self._right_anchor_world, dtype=np.float64).copy()
            self._swing_goal_world = self._proposed_step_world(intent, "right", step_index=phase.step_index)

    def _foot_state_for_side(
        self,
        side: str,
        phase: PhaseState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        anchor = self._left_anchor_world if side == "left" else self._right_anchor_world
        if anchor is None:
            raise RuntimeError("Footstep planner anchors are not initialized")
        if phase.swing_foot != side or self._swing_start_world is None or self._swing_goal_world is None:
            return np.asarray(anchor, dtype=np.float64).copy(), np.zeros(3, dtype=np.float64)

        narrowness = float(phase.metadata.get("narrowness", 0.0))
        gap_severity = float(phase.metadata.get("gap_severity", narrowness))
        swing_height = float(self.cfg.swing_height) * (
            1.0 - float(self.cfg.through_gap_swing_height_scale) * gap_severity
        )
        alpha = float(np.clip(phase.alpha, 0.0, 1.0))
        blend = alpha * alpha * (3.0 - 2.0 * alpha)
        pos = (1.0 - blend) * self._swing_start_world + blend * self._swing_goal_world
        pos = np.asarray(pos, dtype=np.float64)
        pos[2] += swing_height * np.sin(np.pi * alpha)
        vel = (self._swing_goal_world - self._swing_start_world) / max(phase.phase_duration, 1e-6)
        vel = np.asarray(vel, dtype=np.float64)
        vel[2] = 0.0
        return pos, vel

    def _default_anchor_positions(self, intent: TraversalIntent) -> Tuple[np.ndarray, np.ndarray]:
        yaw = float(intent.yaw)
        rot = self._yaw_rotation(yaw)
        envelope_state = self.envelope.evaluate(intent)
        stance_width = float(self.cfg.nominal_stance_width) * (
            1.0 - (1.0 - float(self.cfg.through_gap_stance_scale)) * envelope_state.gap_severity
        )
        left_local = np.asarray([self.cfg.nominal_foot_x, 0.5 * stance_width], dtype=np.float64)
        right_local = np.asarray([self.cfg.nominal_foot_x, -0.5 * stance_width], dtype=np.float64)
        left_xy = intent.planar_position + rot @ left_local
        right_xy = intent.planar_position + rot @ right_local
        left = np.asarray([left_xy[0], left_xy[1], self._left_nominal_z], dtype=np.float64)
        right = np.asarray([right_xy[0], right_xy[1], self._right_nominal_z], dtype=np.float64)
        return left, right

    def _proposed_step_world(self, intent: TraversalIntent, side: str, *, step_index: int = 0) -> np.ndarray:
        yaw = float(intent.yaw)
        rot = self._yaw_rotation(yaw)
        envelope_state = self.envelope.evaluate(intent)
        startup_scale = 1.0
        startup_lateral_scale = 1.0
        startup_preview_scale = 1.0
        if int(step_index) < int(self.cfg.startup_small_step_count):
            startup_scale = float(self.cfg.startup_forward_step_scale)
            startup_lateral_scale = float(self.cfg.startup_lateral_step_scale)
            startup_preview_scale = float(self.cfg.startup_preview_scale)
        stance_width = float(self.cfg.nominal_stance_width) * (
            1.0 - (1.0 - float(self.cfg.through_gap_stance_scale)) * envelope_state.gap_severity
        )
        preview = float(self.cfg.footstep_preview_time) * (
            1.0 - (1.0 - float(self.cfg.through_gap_preview_scale)) * envelope_state.gap_severity
        ) * startup_preview_scale
        preview_xy = np.asarray(intent.planar_velocity, dtype=np.float64) * preview
        center_bias_world = np.asarray([0.0, -float(intent.planar_position[1])], dtype=np.float64)
        nominal_local = np.asarray(
            [
                self.cfg.nominal_foot_x
                + float(intent.planar_velocity[0]) * preview
                * (1.0 - (1.0 - float(self.cfg.through_gap_forward_step_scale)) * envelope_state.gap_severity)
                * startup_scale,
                0.5 * stance_width if side == "left" else -0.5 * stance_width,
            ],
            dtype=np.float64,
        )
        nominal_xy = (
            intent.planar_position
            + rot @ nominal_local
            + 0.2 * preview_xy
            + float(self.cfg.through_gap_center_bias_gain) * envelope_state.gap_severity * center_bias_world
            - float(self.cfg.through_gap_center_velocity_bias) * envelope_state.gap_severity * np.asarray([0.0, intent.planar_velocity[1]], dtype=np.float64)
        )
        anchor = self._left_anchor_world if side == "left" else self._right_anchor_world
        if anchor is None:
            nominal_z = float(self._left_nominal_z if side == "left" else self._right_nominal_z)
            return np.asarray([nominal_xy[0], nominal_xy[1], nominal_z], dtype=np.float64)

        delta_xy = nominal_xy - np.asarray(anchor[:2], dtype=np.float64)
        local_delta = rot.T @ delta_xy
        local_delta[0] = float(
            np.clip(
                local_delta[0],
                -float(self.cfg.max_step_backward)
                * (1.0 - (1.0 - float(self.cfg.through_gap_forward_step_scale)) * envelope_state.gap_severity)
                * startup_scale,
                float(self.cfg.max_step_forward)
                * (1.0 - (1.0 - float(self.cfg.through_gap_forward_step_scale)) * envelope_state.gap_severity)
                * startup_scale,
            )
        )
        local_delta[1] = float(
            np.clip(
                local_delta[1],
                -float(self.cfg.max_step_lateral)
                * (1.0 - (1.0 - float(self.cfg.through_gap_lateral_step_scale)) * envelope_state.gap_severity)
                * startup_lateral_scale,
                float(self.cfg.max_step_lateral)
                * (1.0 - (1.0 - float(self.cfg.through_gap_lateral_step_scale)) * envelope_state.gap_severity)
                * startup_lateral_scale,
            )
        )
        clamped_xy = np.asarray(anchor[:2], dtype=np.float64) + rot @ local_delta
        nominal_z = float(self._left_nominal_z if side == "left" else self._right_nominal_z)
        return np.asarray([clamped_xy[0], clamped_xy[1], nominal_z], dtype=np.float64)

    @staticmethod
    def _yaw_rotation(yaw: float) -> np.ndarray:
        return np.asarray(
            [
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)],
            ],
            dtype=np.float64,
        )
