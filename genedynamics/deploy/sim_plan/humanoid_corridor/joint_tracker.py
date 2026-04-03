"""
Post-process joint references with smoothing and safety clamps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from genedynamics.deploy.sim_plan.humanoid_corridor.g1_model import G1ModelSpec
from genedynamics.deploy.sim_plan.humanoid_corridor.schema import JointTargets


@dataclass
class JointTrackerConfig:
    smoothing_alpha: float = 0.35
    max_delta_per_step: float = 0.12
    max_joint_speed: float = 6.0
    joint_limit_margin: float = 0.02


class JointReferenceTracker:
    """
    Light-weight post processor before sending references to MuJoCo or hardware.

    Stage 2 should extend this class with torque clipping, support-aware damping,
    and fallback posture logic.
    """

    def __init__(self, model_spec: G1ModelSpec, cfg: Optional[JointTrackerConfig] = None) -> None:
        self.model_spec = model_spec
        self.cfg = cfg or JointTrackerConfig()
        self._last_q_ref = self.model_spec.stand_ctrl.copy()

    def reset(self, q_ref: Optional[np.ndarray] = None) -> None:
        self._last_q_ref = self.model_spec.stand_ctrl.copy() if q_ref is None else np.asarray(q_ref, dtype=np.float64).copy()

    def filter(self, targets: JointTargets, *, dt: float) -> JointTargets:
        dt = float(max(dt, 1e-6))
        q_raw = np.asarray(targets.q_ref, dtype=np.float64).reshape(-1)
        q_ref = self._blend(q_raw)
        q_ref = self._limit_delta(q_ref, dt)
        q_ref = self.model_spec.clip_to_joint_limits(q_ref, margin=float(self.cfg.joint_limit_margin))
        qd_ref = (q_ref - self._last_q_ref) / dt
        qd_ref = np.clip(qd_ref, -float(self.cfg.max_joint_speed), float(self.cfg.max_joint_speed))
        self._last_q_ref = q_ref.copy()
        meta = dict(targets.metadata)
        meta["tracker"] = "joint_reference_tracker"
        return JointTargets(
            q_ref=q_ref,
            qd_ref=qd_ref,
            tau_ff=targets.tau_ff,
            metadata=meta,
        )

    def _blend(self, q_ref: np.ndarray) -> np.ndarray:
        alpha = float(np.clip(self.cfg.smoothing_alpha, 0.0, 1.0))
        if alpha <= 0.0:
            return q_ref.copy()
        return (1.0 - alpha) * q_ref + alpha * self._last_q_ref

    def _limit_delta(self, q_ref: np.ndarray, dt: float) -> np.ndarray:
        max_delta = min(float(self.cfg.max_delta_per_step), float(self.cfg.max_joint_speed) * dt)
        delta = np.clip(q_ref - self._last_q_ref, -max_delta, max_delta)
        return self._last_q_ref + delta

