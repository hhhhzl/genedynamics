"""
Footstep planning for humanoid corridor traversal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from genedynamics.deploy.followers.common.traversal_intent import TraversalIntent
from genedynamics.deploy.followers.humanoid.task_spec import ContactPhase, PhaseState


@dataclass
class FootstepPlannerConfig:
    nominal_stance_width: float = 0.24
    nominal_foot_x: float = 0.02
    swing_height: float = 0.05
    footstep_preview_time: float = 0.35
    max_step_forward: float = 0.40
    max_step_backward: float = 0.10
    max_step_lateral: float = 0.14


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
    Convert traversal intent plus contact phase into footstep targets.
    """

    def __init__(self, cfg: Optional[FootstepPlannerConfig] = None) -> None:
        self.cfg = cfg or FootstepPlannerConfig()
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
        *,
        dt: float,
    ) -> FootstepPlan:
        if self._left_anchor_world is None or self._right_anchor_world is None:
            self._left_anchor_world, self._right_anchor_world = self._default_anchor_positions(intent)

        if phase.metadata.get("transitioned", False):
            self._handle_transition(intent, phase)

        left_pos, left_vel = self._foot_state_for_side("left", phase, dt)
        right_pos, right_vel = self._foot_state_for_side("right", phase, dt)
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
            },
        )

    def _handle_transition(self, intent: TraversalIntent, phase: PhaseState) -> None:
        if phase.phase == ContactPhase.DOUBLE_SUPPORT and self._active_swing_foot and self._swing_goal_world is not None:
            if self._active_swing_foot == "left":
                self._left_anchor_world = self._swing_goal_world.copy()
            else:
                self._right_anchor_world = self._swing_goal_world.copy()
            self._active_swing_foot = None
            self._swing_start_world = None
            self._swing_goal_world = None
            return

        if phase.phase == ContactPhase.LEFT_SWING:
            self._active_swing_foot = "left"
            self._swing_start_world = np.asarray(self._left_anchor_world, dtype=np.float64).copy()
            self._swing_goal_world = self._proposed_step_world(intent, "left")
        elif phase.phase == ContactPhase.RIGHT_SWING:
            self._active_swing_foot = "right"
            self._swing_start_world = np.asarray(self._right_anchor_world, dtype=np.float64).copy()
            self._swing_goal_world = self._proposed_step_world(intent, "right")

    def _foot_state_for_side(
        self,
        side: str,
        phase: PhaseState,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        _ = dt
        anchor = self._left_anchor_world if side == "left" else self._right_anchor_world
        if anchor is None:
            raise RuntimeError("Footstep planner anchors are not initialized")
        if phase.swing_foot != side or self._swing_start_world is None or self._swing_goal_world is None:
            return np.asarray(anchor, dtype=np.float64).copy(), np.zeros(3, dtype=np.float64)

        alpha = float(np.clip(phase.alpha, 0.0, 1.0))
        blend = alpha * alpha * (3.0 - 2.0 * alpha)
        pos = (1.0 - blend) * self._swing_start_world + blend * self._swing_goal_world
        pos = np.asarray(pos, dtype=np.float64)
        pos[2] += float(self.cfg.swing_height) * np.sin(np.pi * alpha)
        vel = (self._swing_goal_world - self._swing_start_world) / max(phase.phase_duration, 1e-6)
        vel = np.asarray(vel, dtype=np.float64)
        vel[2] = 0.0
        return pos, vel

    def _default_anchor_positions(self, intent: TraversalIntent) -> Tuple[np.ndarray, np.ndarray]:
        yaw = float(intent.yaw)
        rot = np.asarray(
            [
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)],
            ],
            dtype=np.float64,
        )
        left_local = np.asarray([self.cfg.nominal_foot_x, 0.5 * self.cfg.nominal_stance_width], dtype=np.float64)
        right_local = np.asarray([self.cfg.nominal_foot_x, -0.5 * self.cfg.nominal_stance_width], dtype=np.float64)
        left_xy = intent.planar_position + rot @ left_local
        right_xy = intent.planar_position + rot @ right_local
        left = np.asarray([left_xy[0], left_xy[1], self._left_nominal_z], dtype=np.float64)
        right = np.asarray([right_xy[0], right_xy[1], self._right_nominal_z], dtype=np.float64)
        return left, right

    def _proposed_step_world(self, intent: TraversalIntent, side: str) -> np.ndarray:
        yaw = float(intent.yaw)
        rot = np.asarray(
            [
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)],
            ],
            dtype=np.float64,
        )
        preview = float(self.cfg.footstep_preview_time)
        preview_xy = np.asarray(intent.planar_velocity, dtype=np.float64) * preview
        nominal_local = np.asarray(
            [
                self.cfg.nominal_foot_x + float(intent.planar_velocity[0]) * preview,
                0.5 * self.cfg.nominal_stance_width if side == "left" else -0.5 * self.cfg.nominal_stance_width,
            ],
            dtype=np.float64,
        )
        nominal_xy = intent.planar_position + rot @ nominal_local + 0.2 * preview_xy
        anchor = self._left_anchor_world if side == "left" else self._right_anchor_world
        if anchor is None:
            nominal_z = float(self._left_nominal_z if side == "left" else self._right_nominal_z)
            return np.asarray([nominal_xy[0], nominal_xy[1], nominal_z], dtype=np.float64)

        delta_xy = nominal_xy - np.asarray(anchor[:2], dtype=np.float64)
        local_delta = rot.T @ delta_xy
        local_delta[0] = float(np.clip(local_delta[0], -self.cfg.max_step_backward, self.cfg.max_step_forward))
        local_delta[1] = float(np.clip(local_delta[1], -self.cfg.max_step_lateral, self.cfg.max_step_lateral))
        clamped_xy = np.asarray(anchor[:2], dtype=np.float64) + rot @ local_delta
        nominal_z = float(self._left_nominal_z if side == "left" else self._right_nominal_z)
        return np.asarray([clamped_xy[0], clamped_xy[1], nominal_z], dtype=np.float64)
