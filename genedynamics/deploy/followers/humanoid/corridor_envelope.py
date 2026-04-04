"""
Corridor-specific envelope heuristics shared by humanoid follower modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genedynamics.deploy.followers.common.traversal_intent import TraversalIntent


@dataclass
class CorridorEnvelopeConfig:
    nominal_height: float = 0.75
    minimum_height: float = 0.55
    nominal_width: float = 0.40
    torso_width_gain: float = (0.40 - 0.22) / 1.57
    arm_width_gain: float = 0.05
    through_gap_narrowness_threshold: float = 0.30
    torso_yaw_through_gap_threshold: float = 0.30
    arm_tuck_through_gap_threshold: float = 0.35
    crouch_through_gap_threshold: float = 0.20
    through_gap_control_floor: float = 0.60


@dataclass
class CorridorEnvelopeState:
    crouch_ratio: float
    effective_width: float
    narrowness: float
    gap_severity: float
    through_gap_mode: bool


class CorridorEnvelopeHeuristics:
    def __init__(self, cfg: CorridorEnvelopeConfig | None = None) -> None:
        self.cfg = cfg or CorridorEnvelopeConfig()

    def evaluate(self, intent: TraversalIntent) -> CorridorEnvelopeState:
        crouch_ratio = self._crouch_ratio(float(intent.body_height))
        avg_tuck = 0.5 * (float(intent.left_arm_tuck) + float(intent.right_arm_tuck))
        torso_ratio = min(abs(float(intent.torso_yaw)) / max(float(self.cfg.torso_yaw_through_gap_threshold), 1e-6), 1.0)
        tuck_ratio = min(avg_tuck / max(float(self.cfg.arm_tuck_through_gap_threshold), 1e-6), 1.0)
        crouch_mode_ratio = min(crouch_ratio / max(float(self.cfg.crouch_through_gap_threshold), 1e-6), 1.0)
        narrowness = float(np.clip(max(torso_ratio, tuck_ratio, crouch_mode_ratio), 0.0, 1.0))
        effective_width = float(
            self.cfg.nominal_width
            - self.cfg.torso_width_gain * abs(float(intent.torso_yaw))
            - self.cfg.arm_width_gain * (float(intent.left_arm_tuck) + float(intent.right_arm_tuck))
        )
        through_gap_mode = bool(
            narrowness >= float(self.cfg.through_gap_narrowness_threshold)
            or abs(float(intent.torso_yaw)) >= float(self.cfg.torso_yaw_through_gap_threshold)
            or avg_tuck >= float(self.cfg.arm_tuck_through_gap_threshold)
            or crouch_ratio >= float(self.cfg.crouch_through_gap_threshold)
        )
        gap_severity = float(
            np.clip(
                max(
                    narrowness,
                    float(self.cfg.through_gap_control_floor) if through_gap_mode else narrowness,
                ),
                0.0,
                1.0,
            )
        )
        return CorridorEnvelopeState(
            crouch_ratio=crouch_ratio,
            effective_width=effective_width,
            narrowness=narrowness,
            gap_severity=gap_severity,
            through_gap_mode=through_gap_mode,
        )

    def _crouch_ratio(self, height: float) -> float:
        denom = max(float(self.cfg.nominal_height) - float(self.cfg.minimum_height), 1e-6)
        return float(np.clip((float(self.cfg.nominal_height) - float(height)) / denom, 0.0, 1.0))
