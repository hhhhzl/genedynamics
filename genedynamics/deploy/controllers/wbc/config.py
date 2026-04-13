"""Whole-body controller configuration.

The legacy ``G1WBCTaskStackConfig`` was a flat 92-field dataclass; this
module replaces it with four semantically grouped sub-configs gathered
under :class:`WBCConfig`. The grouping makes it possible to tune one aspect
of the controller (e.g. only the QP solver, only the friction cone bounds)
without scrolling past unrelated knobs.

Groups:

* :class:`TaskWeightsConfig` — least-squares weights of every task in the
  objective stack, including the through-gap and single-support phase
  scales applied on top of the base weight.
* :class:`TaskGainsConfig`   — Kp/Kd PD gains for every task plus the
  posture-bias gains used to shape the lower-body reference.
* :class:`LimitsConfig`      — clipping bounds, friction-cone parameters,
  joint/torque/acceleration limits, and the QP regularization scalar.
* :class:`SolverConfig`      — QP solver selection, iteration budgets, and
  tolerance for the multi-method fallback chain.

Defaults reproduce the legacy values exactly so the new controller is a
drop-in replacement at the parameter level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "TaskWeightsConfig",
    "TaskGainsConfig",
    "LimitsConfig",
    "SolverConfig",
    "WBCConfig",
]


# ---------------------------------------------------------------------------
# Task weights
# ---------------------------------------------------------------------------


@dataclass
class TaskWeightsConfig:
    """Weights of each task in the least-squares objective.

    The total weight applied to a task is::

        base * through_gap_scale(narrowness) * support_scale(num_contacts)

    where the base lives here, ``through_gap`` interpolates from 1.0 at
    ``narrowness=0`` to the configured scale at ``narrowness=1``, and
    ``support`` switches between the configured single-support scale (when
    ``num_contacts == 1``) and ``1.0`` (double support / flight).
    """

    # Base weights
    contact: float = 80.0
    com: float = 8.0
    pelvis: float = 7.0
    torso: float = 6.0
    swing_foot: float = 8.0
    swing_foot_orientation: float = 2.5
    lower_body_posture: float = 1.2
    waist_posture: float = 1.6
    arm_posture: float = 1.2
    posture: float = 0.5

    # Phase scaling — through narrow gap
    through_gap_com_scale: float = 1.35
    through_gap_pelvis_scale: float = 1.25
    through_gap_torso_scale: float = 2.20
    through_gap_swing_scale: float = 0.70
    through_gap_arm_scale: float = 2.40
    through_gap_waist_scale: float = 2.00
    through_gap_lower_body_scale: float = 1.40

    # Phase scaling — single support
    single_support_com_scale: float = 1.20
    single_support_pelvis_scale: float = 1.10
    single_support_torso_scale: float = 0.50
    single_support_swing_scale: float = 0.16
    single_support_arm_scale: float = 0.03
    single_support_waist_scale: float = 0.05
    single_support_lower_body_scale: float = 0.18


# ---------------------------------------------------------------------------
# Task gains
# ---------------------------------------------------------------------------


@dataclass
class TaskGainsConfig:
    """Kp / Kd gains for each task and the lower-body posture biases."""

    # Contact (support feet acceleration tracking)
    contact_position_kp: float = 8.0
    contact_position_kd: float = 6.0
    contact_orientation_kp: float = 18.0
    contact_orientation_kd: float = 6.0

    # CoM
    com_kp: float = 45.0
    com_kd: float = 14.0

    # Body orientation
    pelvis_orientation_kp: float = 40.0
    pelvis_orientation_kd: float = 10.0
    torso_orientation_kp: float = 36.0
    torso_orientation_kd: float = 9.0

    # Swing foot
    swing_foot_position_kp: float = 20.0
    swing_foot_position_kd: float = 6.0
    swing_foot_orientation_kp: float = 8.0
    swing_foot_orientation_kd: float = 3.0

    # Posture
    posture_kp: float = 18.0
    posture_kd: float = 5.0
    waist_kp: float = 24.0
    waist_kd: float = 6.0
    arm_kp: float = 14.0
    arm_kd: float = 4.0

    # Lower-body posture biasing (used by the lower_body_target builder)
    body_height_nominal: float = 0.75
    body_height_min: float = 0.55
    crouch_hip_pitch_gain: float = 0.35
    crouch_knee_gain: float = 0.95
    crouch_ankle_pitch_gain: float = -0.45
    pelvis_forward_hip_pitch_gain: float = 0.0
    pelvis_lateral_hip_roll_gain: float = 0.50


# ---------------------------------------------------------------------------
# Limits & regularization
# ---------------------------------------------------------------------------


@dataclass
class LimitsConfig:
    """Clipping bounds, friction cone, and QP regularization."""

    # Contact tracking error clips
    contact_position_error_clip: float = 0.01
    contact_orientation_error_clip: float = 0.10
    contact_accel_limit: float = 6.0
    contact_angular_accel_limit: float = 8.0

    # Friction pyramid
    friction_coeff: float = 0.60
    lambda_min_normal: float = 20.0
    lambda_max_normal: float = 550.0
    lambda_max_tangent: float = 220.0
    lambda_max_moment_roll: float = 70.0
    lambda_max_moment_pitch: float = 70.0
    lambda_max_moment_yaw: float = 45.0
    lambda_normal_target: float = 35.0

    # Lambda regularization (in objective)
    lambda_normal_weight: float = 0.05
    lambda_tangent_weight: float = 0.02
    lambda_moment_weight: float = 0.01
    lambda_weight: float = 1e-3
    ddq_weight: float = 1e-4

    # Solution clipping
    max_joint_accel: float = 40.0
    max_base_accel: float = 20.0
    max_qd_ref: float = 8.0
    max_q_step: float = 0.12

    # Torque envelope (multiplier on the spec's actuator_forcerange)
    torque_limit_scale: float = 0.85

    # QP regularization (added to H diagonal)
    qp_regularization: float = 1e-8


# ---------------------------------------------------------------------------
# Solver options
# ---------------------------------------------------------------------------


@dataclass
class SolverConfig:
    """Selection and tuning of the QP solver fallback chain."""

    # Primary: full-space OSQP via core/constraints/solvers
    use_osqp: bool = True
    osqp_maxiter: int = 4000
    osqp_polish: bool = True
    osqp_verbose: bool = False
    osqp_accept_constraint_tol: float = 5e-4

    # Secondary: scipy SLSQP on the reduced (null-space) problem
    use_slsqp: bool = True
    slsqp_maxiter: int = 120

    # Tertiary: scipy trust-constr (slower but more robust)
    use_trust_constr: bool = False
    trust_constr_maxiter: int = 80

    # Quaternary: trust-constr repair pass when violation persists
    use_trust_constr_repair: bool = False
    trust_constr_repair_maxiter: int = 40
    trust_constr_repair_violation_threshold: float = 1e-3

    # Single-support gets a beefier trust-constr budget
    single_support_trust_constr_maxiter: int = 160

    # Active-set refinement and projection repair
    active_set_refine_iters: int = 8
    projection_repair_iters: int = 40

    # Acceptance tolerance: any solution with eq_residual ≤ this AND
    # ineq_violation ≤ this is accepted as optimal.
    constraint_tol: float = 1e-6


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class WBCConfig:
    """Top-level whole-body controller config.

    Use the four sub-configs to tune behavior::

        cfg = WBCConfig()
        cfg.weights.swing_foot = 12.0
        cfg.gains.com_kp = 60.0
        cfg.limits.friction_coeff = 0.7
        cfg.solver.use_trust_constr = True
    """

    weights: TaskWeightsConfig = field(default_factory=TaskWeightsConfig)
    gains: TaskGainsConfig = field(default_factory=TaskGainsConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
