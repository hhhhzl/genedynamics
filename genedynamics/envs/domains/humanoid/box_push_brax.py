"""Humanoid contact-manifold box pushing to a line (idea.txt Exp II).

H1 humanoid pushes a RIGID box to a goal LINE (x = box_goal_x) from BEHIND (the rear
face), keeping hand-box contact, balance, and friction feasibility. The box is a
PLANAR 3-DoF body (x-slide + y-slide + yaw-hinge, scene mjx_scene_h1_box_push.xml) so
it can be straight-pushed (L1) and rotated to clear an obstacle (L3). Box shape/size +
friction vary (DR); the box stays RIGID (no deformation).

Levels:
  * push_to_line (L1): robot stands (fixed double-support), hand pushes the box straight
    to the line; box y/yaw held ~0 (alignment). FIRST milestone.
  * unjam (L3): a wall blocks the straight path; the box must be rotated / pushed obliquely
    (contact-face j active, box yaw freed) to reach the line. (built next)
  * push_walk (L2): robot WALKS while pushing (far line). (deferred — reuses the H1 gait)

Layered primitive (idea.txt §2.1; NOT the 26D full setting — that is the general
formulation / future). The active primitive is a CONTACT-SEMANTIC manifold
primitive, not 19 joint references:

    u^hum = (j, a, b, S_hand, F_n^d)              -> 12D   (H1 / H2 / H4-A)
    u^hum = (v_base, j, a, b, S_hand, F_n^d)      -> 15D   (H4-B, ``use_base``)

  * j: relaxed one-hot over contact faces {rear, left, right} (front omitted);
  * (a,b) ∈ [0,1]² local face coordinates;
  * S_hand: log-SPD 3×3 hand stiffness, K_hand = exp(S_hand);
  * F_n^d: desired hand normal (push) force;
  * v_base: small base-velocity command (H4-B), a posture lean.

Simplified π_low (NOT a full WBC — that is the future extension): fixed
double-support stance held by joint PD, the RIGHT arm driven by a Cartesian
hand impedance toward the selected box-face contact point, balance kept by the
stance PD + a small base lean. Motors are torque actuators, so π_low emits joint
torques directly.

``constraint_residual`` exposes the full eq:humanoid_manifold (clean / mjx-kin
state, no mjx backprop): h_box, h_hand, h_hand_R, h_foot (eq) + g_bal, g_fric,
g_tip (ineq), feeding the soft-feasibility (AL) seam. The general ``manifold_residual``
/ ``manifold_geometry`` hooks expose the clean-state constant-stiffness + force-target
manifold (primitive-only); the physics-coupled contact / balance go through the AL.

Imports guarded (brax optional on fedguide).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

try:
    import mujoco
    from brax import math as brax_math
    from brax.envs.base import State
    from mujoco.mjx._src import support as _mjx_support
    BRAX_AVAILABLE = True
except Exception:  # pragma: no cover
    BRAX_AVAILABLE = False
    State = object

from genedynamics.core.control.stiffness import PrimitiveSpec, stiffness_log_to_pd
from genedynamics.core.control.humanoid_contact import (
    HumanoidWholeBodyController,
    hand_contact_wrench,
)
from genedynamics.robots import RobotBinding, get_robot_registry
from genedynamics.robots.profile import (
    BIPED, FLOATING_BASE, TORQUE_CONTROL, TWO_FEET, TWO_HANDS,
    WHOLE_BODY_CONTROL,
)
from genedynamics.envs.legged_brax_base import (
    BaseEnvConfig,
    HumanoidTaskEnv,
    get_foot_step,
    global_to_body_velocity,
)


H1_RELIABILITY_SCHEMA = {
    "version": 4,
    "task": "humanoid_box_push",
    "realization": "finite_hand_acquisition_measured_support_fall_priority_success_absorbing",
    "feature_names": [
        "box_progress", "box_goal_distance", "box_lateral_offset", "box_yaw",
        "hand_force_ratio", "wall_force_ratio", "nonhand_force_ratio",
        "support_offset_ratio", "contact_acquired", "unjam_released",
        "force_integrator_ratio", "corridor_clearance_ratio",
        "translation_action_rms", "stiffness_action_rms", "mean_force_action",
        "max_force_action", "first_action_delta_rms", "action_roughness",
        "mean_semantic_action_0", "mean_semantic_action_3",
        "mean_semantic_action_4", "walk_action_rms", "walk_action_max", "is_walk",
    ],
    "risk_names": ["force_violation", "invalid_contact_or_fall", "balance", "force_mae"],
    "state_feature_count": 12,
    "support_state_feature_count": 12,
    "probability_risk_count": 2,
    "classification_probabilities": True,
    "feature_time": "pre_transition_state_and_complete_dense_candidate",
    "horizon": "Hsample_plus_one_post_transitions",
    "risk_definitions": [
        "fixed_tasks:max_horizon,p4_receding:executed_interval_plus_shifted_backup(initial_and_post_substeps)(hand_force_gt_fmax)",
        "fixed_tasks:max_horizon,p4_receding:executed_interval_plus_shifted_backup(initial_and_post_substeps)(nonhand_force_gt_0.5N_or_torso_up_lt_0_or_height_lt_0.6_reset_height)",
        "fixed_tasks:horizon_tail_balance,p4_receding:max(executed_interval_plus_shifted_backup_balance,normal_horizon_terminal_recovery_core);emergency_uses_physical_balance_only",
        "mean_endpoint_abs_hand_force_minus_time_ramped_tapered_benchmark_over_max_ftarget_1",
    ],
    "physics_sampling": "post_each_fast_force_mjx_substep_no_endpoint_replication",
    "missing_physics_coverage": "active_transition_unknown_not_a_positive_training_label",
    "safe_gate": "first_three_physical_risk_heads_le_zero",
    "execution_scope": "normal_only_exclude_any_unload_transition_in_candidate_window",
    "execution_context_version": 1,
    "initial_risk": "instantaneous_decision_state_not_previous_transition_envelope",
    "termination": "substep_fall_priority_success_absorbing_retain_actual_completion_exclude_unexecuted_padding",
}

_STIFF_D = 3                             # 3×3 Cartesian hand stiffness (task geometry, robot-independent)


@dataclass
class HumanoidBoxPushConfig(BaseEnvConfig):
    robot: str = "h1"               # profile id; task validates required humanoid capabilities
    gait: str = "jog"               # backward-compatible reward reference for push_walk
    level: str = "push_to_line"     # push_to_line(L1) | heavy_dr | unjam(L3)  (push_walk L2 = later)
    # Force control runs at the same 50 Hz control rate (dt=0.02) but needs five 4 ms physics
    # substeps to resolve hand-box impacts without the 20 ms contact-force spikes.
    timestep: float = 0.004
    use_base: bool = False          # H4-B: +v_base -> 15D primitive
    dr_seed: int = 0                # H2 domain-randomization draw
    push_dist: float = 0.10         # fixed-stance H1 reach; longer pushes belong to push_walk
    box_half: float = 0.55          # crate half-extent (~DIAL-size; top reaches the H1 hand height)
    box_mass: float = 8.0           # crate mass kg — light enough the right arm pushes it (DR varies)
    # HAND push: the contact point sits on the crate's upper face at the H1 hand (elbow_link) reach
    # height ~1.05 m (near shoulder) so the SHORT high H1 arm can meet it WITHOUT leaning the body.
    hand_push_height: float = 1.05
    contact_band: float = 0.12
    contact_edge_margin: float = 0.04  # keep DR contact targets inside the actual face
    box_geom_friction: float = 0.6  # hand-box surface (geom) friction
    box_shape: str = "box"          # box geom type (cylinder/sphere = geom-type DR, later)
    goal_eps: float = 0.02          # success tolerance on box x reaching the line
    # planar-box slide frictionloss (x = push axis; y/yaw = drift resistance keeping the
    # straight-push levels aligned), aligned to the ~50N the H1 arm delivers.
    box_frictionloss: float = 15.0  # below the 30N target, above passive contact force so the box can stop
    unjam_frictionloss: float = 38.0  # unjam: jammed straight, needs rotation/oblique push
    unjam_initial_yaw: float = 0.078  # ~1 mm corridor interference, stable physical jam
    unjam_corridor_half_width: float = 0.59
    unjam_release_clearance: float = 0.005  # hysteretic switch to rear-centre contact
    unjam_yaw_eps: float = 0.02
    y_frictionloss: float = 20.0    # box y-slide resistance (lateral drift)
    yaw_frictionloss: float = 8.0   # box yaw resistance (rotation; L3 lowers it to allow turning)
    support_radius: float = 0.25    # balance: CoM xy within this of feet center
    hand_offset: float = 0.21       # two-hand lateral spacing (near the H1 home-hand spacing)
    # L1 is a fixed-base/short-reach force-control task: H1 is already nearly
    # fully extended at home, so use a small positive pre-contact clearance.
    # DIAL-style walk-and-push may explicitly override this to ~0.07 m.
    approach_gap: float = 0.015
    # box domain randomization (mass / size / friction) — heavy_dr draws these
    h2_mass_range: tuple = (5.0, 30.0)
    # Keep the vertical rear face inside the fixed-stance H1 arm workspace;
    # OOD varies size, mass and friction without turning the task unreachable.
    h2_size_range: tuple = (0.48, 0.55)
    # hand stiffness (3×3 log-SPD): K_hand = exp(s_ref_diag) ≈ 55 N/m — stiff enough the arm
    # EXTENDS to the box face against gravity WITHOUT leaning the body (a soft hand can only reach
    # by leaning forward, which tips the H1). The planner modulates S_hand around this.
    s_ref_diag: float = 4.0
    s_scale: float = 1.0
    stiffness_mode: str = "log_spd"     # log_spd | euclid | fixed | none
    # hand impedance / push force
    d_damp: float = 5.0
    f_min: float = 0.0
    f_max: float = 60.0
    f_target: float = 30.0
    # Protocol switch for the P1 force-servo gate.  The experiment owns the
    # requested force in P1, so a planner may not lower it to improve safety.
    fixed_force_target: bool = False
    # P1 is a force-regulation experiment, not a box-to-line task.  Its
    # episode-level pass is evaluated from the recorded 4 ms force tape: the
    # force must rise, remain inside this terminal band for the requested
    # hold, and satisfy the shared physical safety margins throughout.
    force_step_rise_fraction: float = 0.90
    force_step_band_fraction: float = 0.10
    force_step_band_absolute: float = 1.0
    force_step_hold_time: float = 0.20
    # P1 also owns the rear-face center contact location.  This isolates the
    # stiffness-chart geometry; P2+ unlock contact-point selection.
    fixed_contact_target: bool = False
    face_logit_temperature: float = 0.10  # bounded logits express effectively discrete face charts
    unjam_face_logit_bias: tuple = (1.0, -1.0, -1.0)
    unjam_contact_a_bias: float = 0.5
    contact_acquire_force: float = 0.5  # measured hand-box force that starts the force ramp
    contact_acquire_gap: float = 0.02   # or enter this normal-distance approach band
    approach_time: float = 0.5          # ramp Cartesian hand target from reset pose to box surface
    force_ramp_time: float = 0.5        # seconds from first contact to full commanded force
    kp_force: float = 0.3               # fast proportional feedback on measured total hand force
    ki_force: float = 1.0               # per-physics-substep force-integral gain
    force_int_max: float = 30.0         # anti-windup clamp (N)
    # Total two-hand outward wrench used only after the first zero-wrench P4
    # UNLOAD interval.  Delaying it by one control period avoids an impulsive
    # switch at compressed contact while preventing the walking body from
    # carrying passive hands back into the box on the recursive successor.
    emergency_retract_force: float = 20.0
    # P4 scores this small robot-owned knee-residual bank with the same
    # two-interval physical certificate used for deployment.  A single fixed
    # flexion direction is not valid across gait phases: it can clear an ankle
    # at one state but drive the opposite knee into the crate at another.
    emergency_knee_residual_levels: tuple = (1.0, 0.0, -0.5, -1.0)
    # Number of real 20 ms intervals covered by the task-owned emergency and
    # receding-incumbent viability certificate. Fixed-stance tasks retain the
    # historical two-step default; P4 opts into three in its suite contract.
    emergency_backup_steps: int = 2
    # Minimum number of actually committed UNLOAD intervals before a P4
    # NORMAL recovery may be considered.  One preserves the historical
    # immediate-exit behavior; larger values provide a short contact-release
    # dwell whose every interval is still replanned and revalidated.
    emergency_min_dwell_steps: int = 1
    # Optional P4 recovery clock. On entry to UNLOAD, search this many previous
    # external-DIAL frames for the closest measured planner-joint phase and
    # hold the selected phase while UNLOAD is committed. The benchmark
    # step/force/success clocks are never changed.
    emergency_reference_rewind_steps: int = 0
    fast_force_loop: bool = True        # 200Hz servo under the 50Hz planner
    hand_contact_solref: tuple = (0.10, 1.0)  # explicit hand-box pair compliance
    wall_contact_solref: tuple = (0.10, 1.0)  # explicit P3 box-wall compliance
    mu_hand: float = 0.6            # hand-box Coulomb friction (g_fric)
    v_base_scale: float = 0.15      # H4-B base-lean command scale
    # H2 domain randomization ranges
    h2_friction_range: tuple = (0.3, 1.0)
    h2_boxfric_range: tuple = (15.0, 40.0)   # all unjam-able by the ~50N arm push
    # Leg+waist actions are exposed only by push_walk.  Fixed-stance levels use the nominal below;
    # otherwise the optimizer can exploit a knee/ankle impact as an invalid box push.
    leg_scale: float = 0.1          # bounded push_walk leg-target residual (rad per unit action)
    leg_grav_comp: float = 1.0      # cancel leg/waist bias force before stance regulation
    # H1 double-support nominal found by the isolated 100-step stance gate.  The asset home pose
    # falls backward after ~1.2 s; shifting both sagittal hip/ankle targets by -0.2 rad remains
    # upright for the full gate with no non-hand box contact.  G1 overrides these to zero below.
    stance_hip_bias: float = -0.2
    stance_ankle_bias: float = -0.2
    # Fixed-support load compensation.  A hand push at ~1 m height creates a
    # backward tipping moment; distribute the opposing sagittal torque across
    # both ankles.  The gain is the effective N m/N value at the reference
    # load; quadratic scheduling avoids over-bracing light contacts.
    stance_force_ankle_gain: float = 0.0
    stance_force_reference: float = 30.0
    # Closed-loop ankle strategy about the reset pelvis-to-feet offset (N m/m,
    # per ankle).  It corrects feed-forward mismatch without leaning on the box.
    stance_com_ankle_gain: float = 0.0
    stance_com_ankle_damping: float = 0.0
    # Load-dependent sagittal hip target (rad per N).  Used together with the
    # ankle moment to keep the pelvis over the fixed support polygon.
    stance_force_hip_gain: float = 0.0
    # Below this total hand load the nominal double support needs no forward
    # hip lean.  Above it, compensate only the excess tipping load.
    stance_force_hip_deadband: float = 0.0
    # reward weights (eq:humanoid_cost, reduced)
    w_box: float = 5.0
    w_prog: float = 8.0            # drive the box forward (push) — strong, else the planner just balances
    w_upright: float = 2.0          # keep torso vertical (balance)
    w_height: float = 1.5          # keep torso near standing height (don't sink)
    w_alive: float = 1.0           # bonus for not falling
    w_contact: float = 5.0         # drive the hand ONTO the crate face (so the planner reaches + contacts)
    w_force: float = 2.0           # track the ramped desired hand force (normalized)
    w_force_limit: float = 10.0    # suppress hand-force overshoot above f_max
    w_nonhand: float = 10.0        # reject torso/leg/other robot-box collisions
    w_bal: float = 0.5
    w_align: float = 0.5            # straight-push alignment (box y + yaw -> 0; L3 frees yaw)
    w_unjam_yaw: float = 5.0        # align the released box with the corridor
    w_smooth: float = 0.0          # penalize joint velocity (anti-tremor); G1-only (H1 default 0)
    w_stiffness_nominal: float = 0.0  # soft trust around nominal hand log-stiffness
    # DIAL-faithful body-clean terms (DIAL keeps the torso un-twisted + non-spinning so the walk is
    # balanced; without these + with a too-high w_upright the body contorts). DIAL: yaw x0.1, ang_vel x1.
    w_yaw: float = 0.1             # penalize torso YAW (face forward, don't twist)
    w_angvel: float = 0.1          # penalize torso angular velocity (don't spin/wobble)
    # L2 push_walk (DIAL-style gait): the robot STEPS (dynamic balance) while pushing. The gait
    # reward dominates (like DIAL's x5) so the feet track a stepping pattern; vel walks it forward.
    w_gait: float = 0.0            # foot stepping-pattern tracking (L2 only; set in __init__)
    w_vel: float = 0.0             # forward walking velocity (L2 only)
    target_vx: float = 0.25        # L2 forward walk speed (slow push-walk)
    # Opt-in locomotion objective, independently validated from box-force control.
    # Legacy keeps the original term definitions and does not evaluate effort.
    walk_objective_mode: str = "legacy"  # legacy | dial (H1 strict joint-target P4)
    walk_velocity_ramp_time: float = 2.0
    walk_height_target: float = 1.2
    w_walk_effort: float = 0.0
    walk_box_goal_mode: str = "position"  # position | coast (strict H1 joint-target P4)
    walk_force_startup_mode: str = "legacy"  # legacy | synchronized (strict H1 joint-target P4)
    gait_ramp_time: float = 0.5    # smoothly engage the CPG after reset
    gait_cadence: float = 0.8
    gait_swing_frac: float = 0.45
    gait_hip_amp: float = 0.15
    gait_hip_forward_sign: float = 1.0  # legacy; H1's +y hip axis uses -1 for forward swing
    gait_stance_sweep: bool = False
    gait_knee_amp: float = 0.30
    gait_capture_gain: float = 0.25
    gait_capture_lead: float = 0.10
    gait_forward_gain: float = 0.40
    gait_roll_amp: float = 0.15
    gait_roll_capture_gain: float = 1.5
    gait_roll_capture_lead: float = 0.10
    gait_roll_limit: float = 0.30
    # The legacy short-push protocol terminated on box position alone.  The
    # opt-in locomotion protocol also requires the support to move forward.
    walk_success_mode: str = "legacy"  # legacy | locomotion
    walk_min_body_progress: float = 0.20
    walk_min_support_progress: float = 0.15
    walk_min_steps_per_foot: int = 1
    walk_step_min_distance: float = 0.04
    walk_lift_height: float = 0.02
    walk_touchdown_height: float = 0.005
    walk_success_hold_time: float = 0.20
    walk_success_max_box_speed: float = 0.08
    # A dynamic gait cannot satisfy the fixed-stance CoM-at-feet-centre test
    # throughout single-support exchange.  P4 certifies an earlier posture
    # envelope instead; the lower terminal-fall threshold remains unchanged.
    walk_safety_min_torso_up: float = 0.90
    walk_safety_min_height_ratio: float = 0.70
    walk_stop_distance: float = 0.12  # taper task force before the line (locomotion mode only)
    walk_approach_force_floor: float = 0.35  # overcome sliding resistance until entering goal band
    w_walk_progress: float = 0.0     # potential-based supported progress (locomotion mode only)
    w_walk_step: float = 0.0         # newly completed, physically supported forward steps
    w_walk_fall: float = 0.0         # one-shot terminal fall penalty for long-horizon RL
    walk_gait_reference: str = "legacy"  # legacy | cpg
    walk_foot_lift: float = 0.06
    policy_primitive_scale: float = 1.0  # PPO-prior chart only; solver action bounds stay unchanged
    # This task's CPG is a low-level reference, unlike original DIAL which
    # searches joint targets directly.  Its ankle support terms must respect
    # the active stance; these opt-in modes keep the primitive width unchanged.
    walk_leg_control: str = "legacy"  # legacy | support_phase | support_phase_foot_level | joint_target
    # Optional DIAL-produced normalized 11-joint locomotion reference.  A
    # complete 23-D recorded action is accepted as input for convenience, but
    # only its final 11 robot-joint coordinates are consumed: contact
    # stiffness and force remain task/MGA variables and must never inherit an
    # unconstrained unloaded-walking trace.
    walk_joint_reference_path: str = ""
    walk_joint_reference_residual_scale: float = 0.05
    # Policy chart inside the solver's normalized joint residual.  When a
    # verified walking reference is active, ordinary MGA candidates share this
    # bounded local authority instead of searching the complete joint interval.
    policy_joint_reference_residual_scale: float = 0.20
    # L1 lean-and-BRACE (human-like): the CoM must stay over the FEET (not topple onto the box), and
    # a staggered (front-foot-forward) stance is rewarded so the support extends forward -> it can
    # lean to push while braced by the rear leg. G1-L1 only (H1 / L2 default 0).
    w_brace: float = 0.0           # penalize CoM x past the front foot (toppling forward)
    w_stagger: float = 0.0         # reward a forward (staggered) stance so the support spans the lean
    # Null-space posture support for the two-hand push; it does not overwrite the Cartesian wrench.
    arm_posture_kp: float = 12.0       # null-space nominal-push posture (H1; G1 override below)
    arm_posture_kd: float = 1.5
    arm_null_damping: float = 1e-3
    arm_grav_comp: float = 1.0         # cancel arm gravity before regulating contact wrench


class HumanoidBoxPushEnv(HumanoidTaskEnv):
    """Humanoid box pushing/unjamming with the MGA contact-semantic primitive. Drives H1 or G1
    (``config.robot``) through one robot-agnostic whole-body controller (legs/waist = planner WBC
    balance, both hands = Cartesian contact impedance). Reuses the H1 brax base — G1 shares the
    pelvis/torso_link bodies + left_foot/right_foot sites, so only the scene + joint layout differ."""

    # Balance/friction inequalities are measured-state constraints and have
    # zero instantaneous action rows at reset.  Tell ATACOM to use its robust
    # fixed-width QR projection; arm tasks retain the historical SVD path.
    atacom_projection_solver = "damped_qr"

    def make_system(self, config):
        # planar-box scene (RIGID box, 3-DoF x/y/yaw); per-robot scene (H1 / G1 motor-actuated).
        from brax.io import mjcf
        from genedynamics.envs.domains.humanoid.box_push_scene import (
            compose_box_push_model,
        )
        profile = get_robot_registry().get_profile(
            "humanoid", str(getattr(config, "robot", "h1")).lower()
        )
        if profile is None:
            raise ValueError(f"unknown humanoid robot profile: {config.robot}")
        sys = mjcf.load_model(compose_box_push_model(profile))
        # L1/H2 are straight push-to-line tasks.  Keep the richer planar
        # y/yaw motion for unjamming only; otherwise the first validation gate
        # is needlessly mixed with lateral/rotational box control.
        if str(getattr(config, "level", "push_to_line")).lower() != "unjam":
            mj = sys.mj_model
            jr = jnp.asarray(sys.jnt_range)
            jl = np.array(sys.jnt_limited, copy=True)
            for name in ("box_y", "box_yaw"):
                jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT.value, name)
                if jid >= 0:
                    jr = jr.at[jid].set(jnp.array([-1e-6, 1e-6], jr.dtype))
                    jl[jid] = True
            sys = sys.tree_replace({"jnt_range": jr, "jnt_limited": jl})
        return sys.tree_replace({"opt.timestep": config.timestep})

    def __init__(self, config: HumanoidBoxPushConfig | None = None, **kw):
        cfg = config or HumanoidBoxPushConfig(**kw)
        if cfg.walk_force_startup_mode not in {"legacy", "synchronized"}:
            raise ValueError("walk_force_startup_mode must be 'legacy' or 'synchronized'")
        if (not np.isfinite(cfg.force_step_rise_fraction)
                or not 0.0 < cfg.force_step_rise_fraction <= 1.0):
            raise ValueError("force_step_rise_fraction must be in (0, 1]")
        for name in ("force_step_band_fraction", "force_step_band_absolute"):
            value = float(getattr(cfg, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(cfg.force_step_hold_time) or cfg.force_step_hold_time <= 0.0:
            raise ValueError("force_step_hold_time must be finite and positive")
        if (not np.isfinite(cfg.emergency_retract_force)
                or not 0.0 <= cfg.emergency_retract_force <= cfg.f_max):
            raise ValueError("emergency_retract_force must be finite and in [0, f_max]")
        knee_levels = np.asarray(cfg.emergency_knee_residual_levels, dtype=float)
        if (knee_levels.ndim != 1 or not 1 <= knee_levels.size <= 5
                or not np.isfinite(knee_levels).all()
                or np.any(np.abs(knee_levels) > 1.0)):
            raise ValueError(
                "emergency_knee_residual_levels must contain 1--5 finite values in [-1, 1]"
            )
        if (isinstance(cfg.emergency_backup_steps, bool)
                or not isinstance(cfg.emergency_backup_steps, (int, np.integer))
                or not 1 <= int(cfg.emergency_backup_steps) <= 8):
            raise ValueError("emergency_backup_steps must be an integer in [1, 8]")
        if (isinstance(cfg.emergency_min_dwell_steps, bool)
                or not isinstance(
                    cfg.emergency_min_dwell_steps, (int, np.integer)
                )
                or not 1 <= int(cfg.emergency_min_dwell_steps) <= 8):
            raise ValueError(
                "emergency_min_dwell_steps must be an integer in [1, 8]"
            )
        if (isinstance(cfg.emergency_reference_rewind_steps, bool)
                or not isinstance(
                    cfg.emergency_reference_rewind_steps, (int, np.integer)
                )
                or int(cfg.emergency_reference_rewind_steps) < 0):
            raise ValueError(
                "emergency_reference_rewind_steps must be a nonnegative integer"
            )
        if cfg.walk_force_startup_mode == "synchronized" and not (
            cfg.robot.lower() == "h1" and cfg.level.lower() == "push_walk"
            and cfg.walk_leg_control == "joint_target"
            and cfg.walk_success_mode == "locomotion"
        ):
            raise ValueError("synchronized force startup requires H1 joint_target locomotion P4")
        if cfg.walk_box_goal_mode not in {"position", "coast"}:
            raise ValueError("walk_box_goal_mode must be 'position' or 'coast'")
        if cfg.walk_box_goal_mode == "coast" and not (
            cfg.robot.lower() == "h1" and cfg.level.lower() == "push_walk"
            and cfg.walk_leg_control == "joint_target"
            and cfg.walk_success_mode == "locomotion"
        ):
            raise ValueError("coast box goal requires H1 joint_target locomotion P4")
        if cfg.walk_objective_mode not in {"legacy", "dial"}:
            raise ValueError("walk_objective_mode must be 'legacy' or 'dial'")
        if cfg.walk_objective_mode == "dial":
            if not (cfg.robot.lower() == "h1" and cfg.level.lower() == "push_walk"
                    and cfg.walk_leg_control == "joint_target"
                    and cfg.walk_success_mode == "locomotion"):
                raise ValueError("dial walk objective requires H1 joint_target locomotion P4")
            for name in ("walk_velocity_ramp_time", "walk_height_target"):
                value = float(getattr(cfg, name))
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(f"{name} must be finite and positive")
            if not np.isfinite(cfg.target_vx) or cfg.target_vx < 0.0:
                raise ValueError("dial walk target_vx must be finite and nonnegative")
        # G1-specific defaults (stability + geometry): the H1 defaults blow up / mis-fit the smaller,
        # weaker G1. Applied only where the user left the field at its H1 default (explicit values win).
        #  - timestep 0.004: G1 is built for a fine step (native 0.004, implicitfast); 0.02 NaNs.
        #  - leg_scale 0.3 / f_max 40: gentler legs+push — G1's aggressive WBC otherwise spikes contacts.
        #  - box 0.4/12 kg, hand 0.75 m: sized to the shorter G1 + its wrist (right_palm) reach.
        if cfg.robot.lower() == "g1":
            _h1 = HumanoidBoxPushConfig()
            _g1 = dict(timestep=0.004, f_max=40.0, s_ref_diag=3.5, hand_push_height=0.88,
                       box_half=0.5, box_mass=12.0, w_smooth=0.01, w_height=3.0, arm_posture_kp=18.0,
                       arm_posture_kd=2.0, hand_offset=0.23,
                       stance_hip_bias=0.0, stance_ankle_bias=0.0,
                       leg_scale=0.3)   # bounded leg authority (same base for both levels)
            if cfg.level.lower() == "push_walk":   # L2: WALK (reward = gait + forward vel)
                _g1.update(w_gait=30.0, w_vel=1.0, w_upright=2.0, w_bal=0.0, box_frictionloss=8.0)
            else:                                  # L1: STAND + lean-brace (reward = brace + upright)
                _g1.update(w_upright=1.5, w_bal=0.05, box_frictionloss=15.0, w_brace=3.0, w_stagger=1.0)
            for _k, _v in _g1.items():
                if getattr(cfg, _k) == getattr(_h1, _k):
                    setattr(cfg, _k, _v)
        # The absorbing goal receives the no-penalty upper bound of the
        # existing active objective.  Negative/nonfinite penalty weights
        # would invalidate that contract rather than justify a large bonus.
        if any(not np.isfinite(float(value)) or float(value) < 0.0
               for name, value in vars(cfg).items() if name.startswith("w_")):
            raise ValueError("box-push reward weights must be finite and nonnegative")
        if not np.isfinite(cfg.push_dist) or cfg.push_dist < 0.0:
            raise ValueError("push_dist must be finite and nonnegative")
        self._robot_profile = get_robot_registry().get_profile(
            "humanoid", str(cfg.robot).lower()
        )
        if self._robot_profile is None:
            raise ValueError(f"unknown humanoid robot profile: {cfg.robot}")
        self._robot_profile.require({
            FLOATING_BASE, BIPED, TWO_FEET, TWO_HANDS,
            TORQUE_CONTROL, WHOLE_BODY_CONTROL,
        })
        groups = self._robot_profile.joint_groups
        self._right_arm = tuple(groups["right_arm"])
        self._left_arm = tuple(groups["left_arm"])
        self._sag_legs = (
            tuple(groups["left_sagittal_leg"]),
            tuple(groups["right_sagittal_leg"]),
        )
        self._roll_hips = tuple(groups["hip_roll"])
        self._n_planner = len(groups["planner"])
        self._is_walk = str(cfg.level).lower() == "push_walk"
        if cfg.walk_success_mode not in {"legacy", "locomotion"}:
            raise ValueError("walk_success_mode must be 'legacy' or 'locomotion'")
        if cfg.walk_gait_reference not in {"legacy", "cpg"}:
            raise ValueError("walk_gait_reference must be 'legacy' or 'cpg'")
        if cfg.walk_leg_control not in {"legacy", "support_phase", "support_phase_foot_level", "joint_target"}:
            raise ValueError("invalid walk_leg_control")
        if cfg.walk_leg_control == "joint_target" and not self._is_walk:
            raise ValueError("joint_target is only defined for push_walk")
        if cfg.walk_joint_reference_path and not (
            cfg.robot.lower() == "h1" and self._is_walk
            and cfg.walk_leg_control == "joint_target"
            and cfg.walk_success_mode == "locomotion"
        ):
            raise ValueError(
                "walk_joint_reference_path requires H1 joint_target locomotion P4"
            )
        if cfg.emergency_reference_rewind_steps > 0 and not (
            cfg.robot.lower() == "h1" and self._is_walk
            and cfg.walk_leg_control == "joint_target"
            and cfg.walk_success_mode == "locomotion"
            and bool(cfg.walk_joint_reference_path)
        ):
            raise ValueError(
                "emergency reference rewind requires an H1 P4 DIAL reference"
            )
        if cfg.emergency_min_dwell_steps > 1 and not (
            cfg.robot.lower() == "h1" and self._is_walk
            and cfg.walk_leg_control == "joint_target"
            and cfg.walk_success_mode == "locomotion"
            and bool(cfg.walk_joint_reference_path)
        ):
            raise ValueError(
                "emergency dwell requires an H1 P4 DIAL reference"
            )
        if (not np.isfinite(cfg.walk_joint_reference_residual_scale)
                or not 0.0 <= cfg.walk_joint_reference_residual_scale <= 1.0):
            raise ValueError(
                "walk_joint_reference_residual_scale must be in [0, 1]"
            )
        if (not np.isfinite(cfg.policy_joint_reference_residual_scale)
                or not 0.0 < cfg.policy_joint_reference_residual_scale <= 1.0):
            raise ValueError(
                "policy_joint_reference_residual_scale must be in (0, 1]"
            )
        if (not np.isfinite(cfg.policy_primitive_scale)
                or not 0.0 < cfg.policy_primitive_scale <= 1.0):
            raise ValueError("policy_primitive_scale must be in (0, 1]")
        if (not np.isfinite(cfg.walk_safety_min_torso_up)
                or not 0.0 < cfg.walk_safety_min_torso_up < 1.0):
            raise ValueError("walk_safety_min_torso_up must be in (0, 1)")
        if (not np.isfinite(cfg.walk_safety_min_height_ratio)
                or not 0.6 < cfg.walk_safety_min_height_ratio < 1.0):
            raise ValueError("walk_safety_min_height_ratio must be in (0.6, 1)")
        self._walk_requires_locomotion = (
            self._is_walk and cfg.walk_success_mode == "locomotion"
        )
        if self._is_walk and cfg.walk_leg_control != "joint_target":
            # Legacy/residual modes use the CPG as the commanded reference.
            # joint_target leaves locomotion to the planner; gait is reward-only.
            from genedynamics.core.control.bipedal_gait import BipedalGait, GaitParams
            _lg, _rg = self._sag_legs
            self._gait_cpg = BipedalGait(
                _lg, _rg, self._robot_profile.num_actuated, GaitParams(
                cadence=cfg.gait_cadence,
                swing_frac=cfg.gait_swing_frac,
                hip_amp=cfg.gait_hip_amp,
                hip_forward_sign=cfg.gait_hip_forward_sign,
                stance_sweep=cfg.gait_stance_sweep,
                knee_amp=cfg.gait_knee_amp,
                capture_gain=cfg.gait_capture_gain,
                capture_lead=cfg.gait_capture_lead,
                forward_gain=cfg.gait_forward_gain,
                roll_amp=cfg.gait_roll_amp,
                roll_capture_gain=cfg.gait_roll_capture_gain,
                roll_capture_lead=cfg.gait_roll_capture_lead,
                roll_limit=cfg.gait_roll_limit,
            ), hip_roll=self._roll_hips)
        super().__init__(cfg)
        self._robot_binding = RobotBinding.from_mujoco_model(
            self._robot_profile, self.sys.mj_model
        )
        # keep the robot's actuated joints; drop the 3 planar-box dofs (x/y/yaw) at the tail.
        self.physical_joint_range = self.physical_joint_range[
            :self._robot_profile.num_actuated
        ]
        self.joint_range = self.physical_joint_range
        self._bcfg = cfg
        self._face_select = str(cfg.level).lower() == "unjam"   # j active only for unjam
        self._align_yaw = 0.0 if str(cfg.level).lower() == "unjam" else 1.0  # L3 frees box yaw
        mj = self.sys.mj_model
        bid = lambda n: mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY.value, n)
        self._box_idx = bid("box_body")
        self._rhand_body = self._robot_binding.element_ids["right_hand"]
        self._lhand_body = self._robot_binding.element_ids["left_hand"]
        self._box_geom = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM.value, "static_box")
        self._rhand_geom = self._robot_binding.element_ids["right_hand_contactor"]
        self._lhand_geom = self._robot_binding.element_ids["left_hand_contactor"]
        self._floor_geom = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM.value, "floor")
        self._foot_body_ids = jnp.asarray(
            mj.site_bodyid[np.asarray(self._feet_site_id)], jnp.int32
        )
        # The box is a separate root body: only the floating robot subtree
        # determines the nominal support load, including under box-mass OOD.
        self._robot_weight = float(
            mj.body_subtreemass[self._pelvis_idx] * np.linalg.norm(mj.opt.gravity)
        )
        self._wall_geoms = jnp.asarray([
            mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM.value, "unjam_wall_left"),
            mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM.value, "unjam_wall_right"),
        ])
        if min(self._box_geom, self._rhand_geom, self._lhand_geom,
               *[int(x) for x in self._wall_geoms]) < 0:
            raise ValueError("box-push scene is missing a named box/hand collision geom")
        # nominal standing torso height (robot-agnostic): forward-kinematics the home keyframe.
        _d = mujoco.MjData(mj); _d.qpos[:] = mj.keyframe("home").qpos
        mujoco.mj_forward(mj, _d)
        self._torso_z0 = float(_d.xpos[self._torso_idx, 2])
        self._feet_home = jnp.asarray(_d.site_xpos[np.asarray(self._feet_site_id)], jnp.float32)
        self._rhand_home = jnp.asarray(_d.geom_xpos[self._rhand_geom], jnp.float32)
        self._lhand_home = jnp.asarray(_d.geom_xpos[self._lhand_geom], jnp.float32)

        # primitive: pos = [v_base(3)?, j(3), a, b]; stiff = svec(3×3)=6; feed = F_n
        pos_dim = (3 if cfg.use_base else 0) + 3 + 2
        self.spec = PrimitiveSpec(pos_dim=pos_dim, stiff_dim=_STIFF_D, feed_dim=1)
        diag_idx = jnp.cumsum(jnp.arange(_STIFF_D, 0, -1)) - jnp.arange(_STIFF_D, 0, -1)
        self._s_ref = jnp.zeros((self.spec.stiff_width,), jnp.float32).at[diag_idx].set(cfg.s_ref_diag)

        nj = self._robot_profile.num_actuated
        defaults = self._robot_profile.controller_defaults
        kp = jnp.asarray(defaults.get("kp", self._config.kp), jnp.float32)
        kd = jnp.asarray(defaults.get("kd", self._config.kd), jnp.float32)
        self._defaultN = jnp.asarray(self._default_pose[:nj], jnp.float32)
        self._stanceN = self._defaultN
        self._kpN = kp[:nj]
        self._kdN = kd[:nj]
        self._tau_lim = jnp.asarray(self.joint_torque_range[:, 1], jnp.float32)[:nj]
        self._arm_mask = jnp.zeros((nj,), bool).at[jnp.asarray(self._right_arm)].set(True)
        self._left_mask = jnp.zeros((nj,), bool).at[jnp.asarray(self._left_arm)].set(True)
        self._arm_any = self._arm_mask | self._left_mask
        # Planner leg residuals are interpreted around the CPG reference in L2.
        # Keep the physical ranges for diagnostics; the low-level controller
        # never maps a sampler draw directly across the full joint range.
        jr = jnp.asarray(self.physical_joint_range, jnp.float32)
        self._leg_lo = jr[:self._n_planner, 0]
        self._leg_hi = jr[:self._n_planner, 1]
        self._full_leg = False
        # clean two-hand PUSH pose (FK-derived): shoulders forward (pitch -1.0), elbows bent (1.0),
        # palms facing +x at chest height. The arm posture is biased toward this (G1); H1 uses the
        # default pose (arm_posture_kp 0 -> unused).
        arm_push = self._defaultN
        if cfg.robot.lower() == "g1":
            arm_push = arm_push.at[jnp.asarray(self._right_arm)].set(
                jnp.array([-1.0, -0.2, 0., 1.0, 0., 0., 0.], jnp.float32))
            arm_push = arm_push.at[jnp.asarray(self._left_arm)].set(
                jnp.array([-1.0, 0.2, 0., 1.0, 0., 0., 0.], jnp.float32))
        self._arm_push_pose = arm_push
        self._walk_joint_reference = None
        self._walk_joint_reference_sha256 = None
        if cfg.walk_joint_reference_path:
            reference_path = Path(cfg.walk_joint_reference_path)
            if not reference_path.is_absolute():
                reference_path = Path.cwd() / reference_path
            payload_bytes = reference_path.read_bytes()
            payload = json.loads(payload_bytes)
            actions = np.asarray(
                payload.get("actions", payload) if isinstance(payload, dict) else payload,
                dtype=np.float32,
            )
            if actions.ndim != 2 or actions.shape[0] < 2:
                raise ValueError("walk joint reference must be a nonempty 2-D action array")
            if actions.shape[1] == self.spec.total_width + self._n_planner:
                actions = actions[:, self.spec.total_width:]
            if actions.shape[1] != self._n_planner:
                raise ValueError(
                    "walk joint reference width must match the robot planner joints"
                )
            if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1.0 + 1e-6):
                raise ValueError("walk joint reference must contain finite normalized actions")
            self._walk_joint_reference = jnp.asarray(
                np.clip(actions, -1.0, 1.0), jnp.float32
            )
            self._walk_joint_reference_sha256 = hashlib.sha256(
                payload_bytes
            ).hexdigest()

        # box slide frictionloss per level (the box slide is the last dof), aligned to
        # the ~50N the arm delivers; H2 also randomizes the hand friction mu.
        mu, mass, half = cfg.mu_hand, cfg.box_mass, cfg.box_half
        lvl = str(cfg.level).lower()
        if lvl == "heavy_dr":                                    # box DR: mu / slide / mass / size
            ks = jax.random.split(jax.random.PRNGKey(cfg.dr_seed + 7717), 4)
            mu = float(jax.random.uniform(ks[0], minval=cfg.h2_friction_range[0], maxval=cfg.h2_friction_range[1]))
            bf = float(jax.random.uniform(ks[1], minval=cfg.h2_boxfric_range[0], maxval=cfg.h2_boxfric_range[1]))
            mass = float(jax.random.uniform(ks[2], minval=cfg.h2_mass_range[0], maxval=cfg.h2_mass_range[1]))
            half = float(jax.random.uniform(ks[3], minval=cfg.h2_size_range[0], maxval=cfg.h2_size_range[1]))
        elif lvl == "unjam":
            bf = cfg.unjam_frictionloss
        else:                                                    # push_to_line / push_walk
            bf = cfg.box_frictionloss
        self._half = jnp.float32(half)                           # used by _contact_target (synced to geom_size)
        if cfg.robot.lower() == "h1" and lvl == "unjam":
            required_half = abs(float(cfg.hand_offset)) + float(cfg.contact_edge_margin)
            if (not np.isfinite(float(self._half)) or not np.isfinite(required_half)
                    or float(self._half) <= required_half):
                raise ValueError(
                    "H1 unjam requires a finite positive two-hand contact span: "
                    "actual box half-extent must exceed abs(hand_offset) + contact_edge_margin"
                )
        # planar box = 3 tail dofs (x-slide/y-slide/yaw-hinge): x = level slide resistance, y/yaw
        # = drift resistance (L3 lowers yaw). Plus box DR: mass, size (re-seated on floor), friction.
        fl = (self.sys.dof_frictionloss
              .at[-3].set(bf).at[-2].set(cfg.y_frictionloss).at[-1].set(cfg.yaw_frictionloss))
        bm = self.sys.body_mass.at[self._box_idx].set(mass)
        gs = self.sys.geom_size.at[self._box_geom].set(jnp.array([half, half, half], jnp.float32))
        gf = self.sys.geom_friction.at[self._box_geom, 0].set(cfg.box_geom_friction)
        bp = self.sys.body_pos.at[self._box_idx, 2].set(half)    # re-seat box bottom on the floor
        pair_id = lambda n: mujoco.mj_name2id(
            mj, mujoco.mjtObj.mjOBJ_PAIR.value, n
        )
        hand_pairs = jnp.asarray([
            pair_id("box_lhand"), pair_id("box_rhand"),
        ])
        wall_pairs = jnp.asarray([
            pair_id("box_wall_left"), pair_id("box_wall_right"),
        ])
        if min(*[int(x) for x in hand_pairs], *[int(x) for x in wall_pairs]) < 0:
            raise ValueError("box-push scene is missing a named contact pair")
        pair_solref = jnp.asarray(self.sys.pair_solref)
        pair_solref = pair_solref.at[hand_pairs].set(
            jnp.asarray(cfg.hand_contact_solref, jnp.float32))
        pair_solref = pair_solref.at[wall_pairs].set(
            jnp.asarray(cfg.wall_contact_solref, jnp.float32))
        self.sys = self.sys.tree_replace({"dof_frictionloss": fl, "body_mass": bm,
                                          "geom_size": gs, "geom_friction": gf, "body_pos": bp,
                                          "pair_solref": pair_solref})
        if cfg.walk_box_goal_mode == "coast":
            # The MJX arrays, not the unmodified native template, own the load.
            joint = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, "box_x")
            if joint < 0:
                raise ValueError("coast box goal requires a named box_x joint")
            dof = int(self.sys.jnt_dofadr[joint])
            mass = float(self.sys.body_mass[self._box_idx])
            resistance = float(self.sys.dof_frictionloss[dof])
            if (not np.isfinite(mass) or mass <= 0.0
                    or not np.isfinite(resistance) or resistance <= 0.0):
                raise ValueError("coast box goal requires positive finite executed mass and x frictionloss")
            self._walk_coast_deceleration = resistance / mass
        self._acquisition_box = None
        if (str(cfg.robot).lower() == "h1"
                and int(mj.geom_type[self._box_geom]) == int(mujoco.mjtGeom.mjGEOM_BOX)
                and all(int(mj.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
                        for g in (self._rhand_geom, self._lhand_geom))):
            from genedynamics.envs.obstacles.convex import BoxObstacle
            # Use the actual DR dimensions, after updating the physics model.
            # MJX sphere--box contact.dist can be a 1.0 sentinel even at a
            # small positive separation, so it cannot measure this near band.
            self._acquisition_box = BoxObstacle(
                np.zeros(3), np.asarray(self.sys.geom_size[self._box_geom])
            )
        if lvl != "unjam":
            gp = self.sys.geom_pos.at[self._wall_geoms, 1].set(
                jnp.array([50.0, -50.0], jnp.float32))
            # P1/P2/P4 are the original open push-to-line / walk-and-push
            # tasks.  The corridor is P3-only, so remove it from both physics
            # reach and rendering instead of merely parking visible walls a
            # few metres away.
            gr = self.sys.geom_rgba.at[self._wall_geoms, 3].set(0.0)
            self.sys = self.sys.tree_replace({"geom_pos": gp, "geom_rgba": gr})
            # Brax physics consumes the arrays above, while its MuJoCo image
            # renderer reconstructs the scene from ``sys.mj_model``.  Keep the
            # native render model synchronized or P3 walls leak into every GIF.
            wall_ids = np.asarray(self._wall_geoms, dtype=int)
            mj.geom_pos[wall_ids, 1] = np.asarray([50.0, -50.0])
            mj.geom_rgba[wall_ids, 3] = 0.0
        self._box_frictionloss = jnp.float32(bf)
        self._mu = jnp.float32(mu)

        # Place the box from the PHYSICAL hand collision geometry, not from the
        # elbow/wrist body origin.  At home the rear face is approach_gap ahead
        # of the frontmost hand surface for every box size/DR draw.  This also
        # keeps the visual line synchronized with the actual reset goal.
        home = mujoco.MjData(mj)
        home.qpos[:] = np.asarray(self._init_q)
        mujoco.mj_forward(mj, home)
        self._torso_z0 = float(home.xpos[self._torso_idx, 2])
        self._feet_home = jnp.asarray(
            home.site_xpos[np.asarray(self._feet_site_id)], jnp.float32)
        self._stance_com_x0 = jnp.float32(
            home.xpos[self._pelvis_idx, 0] - self._feet_home[:, 0].mean())
        self._rhand_home = jnp.asarray(home.geom_xpos[self._rhand_geom], jnp.float32)
        self._lhand_home = jnp.asarray(home.geom_xpos[self._lhand_geom], jnp.float32)
        hand_front_x = max(float(home.geom_xpos[self._rhand_geom, 0]),
                           float(home.geom_xpos[self._lhand_geom, 0])) + float(
                               self._robot_profile.controller_defaults[
                                   "hand_forward_extent"
                               ]
                           )
        yaw0 = float(cfg.unjam_initial_yaw) if lvl == "unjam" else 0.0
        rear_extent = float(half) * (abs(np.cos(yaw0)) + abs(np.sin(yaw0)))
        box_x0 = hand_front_x + float(cfg.approach_gap) + rear_extent
        box_x_joint = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT.value, "box_x")
        box_qadr = int(mj.jnt_qposadr[box_x_joint])
        self._init_q = self._init_q.at[box_qadr].set(jnp.float32(box_x0))
        box_yaw_joint = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT.value, "box_yaw")
        box_yaw_qadr = int(mj.jnt_qposadr[box_yaw_joint])
        self._init_q = self._init_q.at[box_yaw_qadr].set(jnp.float32(yaw0))
        goal_site = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE.value, "goal_line")
        if goal_site >= 0:
            # Success is expressed by the box-centre target, but a physical
            # push-to-line marker denotes where the leading face stops.  Draw
            # it at the front edge of that terminal box pose so it remains
            # visible instead of passing through the middle of the crate.
            goal_x = jnp.float32(box_x0 + cfg.push_dist + half)
            show_goal_line = not cfg.fixed_force_target and lvl != "unjam"
            site_pos = self.sys.site_pos.at[goal_site, 0].set(goal_x)
            self.sys = self.sys.tree_replace({"site_pos": site_pos})
            # MJX consumes ``self.sys`` while Brax's offline renderer reads the
            # native model retained on ``sys.mj_model``.  Synchronize the
            # position in both; marker RGBA exists only on the native model.
            # P2/P4 show the actual reset-relative terminal face, whereas
            # force-only P1 and geometry-correction P3 have no line objective.
            mj.site_pos[goal_site, 0] = float(goal_x)
            mj.site_rgba[goal_site, 3] = 0.4 if show_goal_line else 0.0
        self._whole_body_controller = HumanoidWholeBodyController(
            binding=self._robot_binding,
            config=self._bcfg,
            default_pose=self._defaultN,
            stance_pose=self._stanceN,
            kp=self._kpN,
            kd=self._kdN,
            torque_limits=self._tau_lim,
            arm_push_pose=self._arm_push_pose,
            right_arm=self._right_arm,
            left_arm=self._left_arm,
            sagittal_legs=self._sag_legs,
            n_planner=self._n_planner,
            pelvis_body_id=self._pelvis_idx,
            feet_site_ids=self._feet_site_id,
            stance_com_x0=self._stance_com_x0,
            is_walk=self._is_walk,
            gait_controller=getattr(self, "_gait_cpg", None),
            joint_limits=self.physical_joint_range,
            planner_joint_bounds=(
                defaults.get("planner_joint_bounds")
                if cfg.walk_leg_control == "joint_target" else None
            ),
            planner_action_reference=self._walk_joint_reference,
            planner_reference_residual_scale=(
                cfg.walk_joint_reference_residual_scale
                if self._walk_joint_reference is not None else None
            ),
        )

    @property
    def action_size(self) -> int:
        # Fixed double-support levels expose only the contact-semantic primitive.  Leg targets are
        # part of the later push_walk controller, not an alternate way to strike the box.
        return self.spec.total_width + (self._n_planner if self._is_walk else 0)

    @property
    def plan_initializer(self):
        """Optional receding-plan initialization in this task's action chart."""
        if self._is_walk and self._bcfg.walk_leg_control == "joint_target":
            return self._initialize_joint_target_plan
        return None

    def _initialize_joint_target_plan(self, state, nodes, *, hold_current=False):
        if getattr(self, "_walk_joint_reference", None) is not None:
            if not hold_current:
                return nodes.at[:, self.spec.total_width:].set(0.0)
            # UNLOAD removes Cartesian hand authority, so continuing the
            # time-indexed walking reference can drive a foot or the passive
            # hands into the box.  Re-express the measured joint pose in the
            # same bounded residual chart on every receding step.  The joint
            # PD derivative term then provides braking without inventing a
            # second locomotion controller or changing the normal reference.
            joints = state.pipeline_state.qpos[
                jnp.asarray(self._robot_binding.qpos_indices[:self._n_planner])
            ]
            current = self._whole_body_controller.planner_action_from_joints(joints)
            index = jnp.clip(
                jnp.asarray(
                    state.info.get(
                        "walk_reference_step", state.info.get("step", 0)
                    ),
                    jnp.int32,
                ),
                0,
                self._walk_joint_reference.shape[0] - 1,
            )
            scale = jnp.float32(self._bcfg.walk_joint_reference_residual_scale)
            residual = jnp.where(
                scale > 0.0,
                (current - self._walk_joint_reference[index])
                / jnp.maximum(scale, jnp.finfo(jnp.float32).eps),
                jnp.zeros_like(current),
            )
            return nodes.at[:, self.spec.total_width:].set(
                jnp.clip(residual, -1.0, 1.0)
            )
        joints = state.pipeline_state.qpos[
            jnp.asarray(self._robot_binding.qpos_indices[:self._n_planner])
        ]
        command = self._whole_body_controller.planner_action_from_joints(joints)
        return nodes.at[:, self.spec.total_width:].set(command)

    @property
    def policy_interface(self):
        """JSON action/observation contract, excluding randomized physical domains.

        The opt-in chart is still 23D for the usual 12D hand primitive, but an
        old residual-action checkpoint is not semantically interchangeable.
        Legacy tasks deliberately retain their previous loading contract.
        """
        if not (self._is_walk and self._bcfg.walk_leg_control == "joint_target"):
            return None
        cfg = self._bcfg
        groups = self._robot_profile.joint_groups
        interface = {
            "schema_version": 1,
            "robot": self._robot_profile.model_id,
            "action_layout": {
                "action_size": int(self.action_size),
                "primitive_width": int(self.spec.total_width),
                "primitive_block_widths": [int(self.spec.pos_dim), int(self.spec.stiff_width),
                                           int(self.spec.feed_dim)],
                "use_base": bool(cfg.use_base),
                "stiffness_mode": cfg.stiffness_mode,
                "planner_joint_names": [self._robot_profile.actuated_joints[i]
                                        for i in groups["planner"]],
                "planner_joint_bounds": np.asarray(
                    self._whole_body_controller.planner_joint_bounds, dtype=float
                ).tolist(),
                "planner_mapping": "affine_minus_one_one_to_joint_bounds",
            },
            "control": {
                "dt": float(self.dt),
                "timestep": float(cfg.timestep),
                "walk_leg_control": cfg.walk_leg_control,
                "leg_grav_comp": float(cfg.leg_grav_comp),
            },
            "observation_layout": {
                "version": "h1_joint_target_qpos_qvel_task21",
                "observation_size": int(self.sys.nq + self.sys.nv + 21),
                "qpos_size": int(self.sys.nq),
                "qvel_size": int(self.sys.nv),
                "task_feature_names": [
                    "box_x", "box_y", "box_yaw", "box_goal_x", "goal_error_x", "mu_hand",
                    "requested_force_over_fmax", "hand_force_over_fmax", "wall_force_over_fmax",
                    "nonhand_force_over_fmax", "pelvis_support_distance_over_radius",
                    "corridor_clearance", "contact_acquired", "unjam_released",
                    "force_integral_over_limit", "level_push_to_line", "level_heavy_dr",
                    "level_unjam", "level_push_walk", "reward_phase_sin", "reward_phase_cos",
                ],
            },
            "task": {
                "level": cfg.level,
                "walk_success_mode": cfg.walk_success_mode,
                "gait": cfg.gait,
                "walk_gait_reference": cfg.walk_gait_reference,
                "walk_safety_envelope": {
                    "min_torso_up": float(cfg.walk_safety_min_torso_up),
                    "min_height_ratio": float(cfg.walk_safety_min_height_ratio),
                    "fixed_stance_proxy_applies": not bool(
                        getattr(
                            self,
                            "_walk_requires_locomotion",
                            self._is_walk and cfg.walk_success_mode == "locomotion",
                        )
                    ),
                },
                "reward_gait_cadence": float(
                    self._gait_params[self._gait][1]
                    if cfg.walk_gait_reference == "legacy" else cfg.gait_cadence
                ),
            },
        }
        if getattr(self, "_walk_joint_reference", None) is not None:
            interface["action_layout"].update({
                "planner_mapping": "dial_reference_plus_bounded_normalized_residual",
                "planner_reference_length": int(self._walk_joint_reference.shape[0]),
                "planner_reference_sha256": self._walk_joint_reference_sha256,
                "planner_reference_residual_scale": float(
                    cfg.walk_joint_reference_residual_scale
                ),
            })
        if cfg.walk_success_mode == "locomotion":
            startup_cap = max(
                float(cfg.approach_time) + float(cfg.force_ramp_time),
                float(cfg.gait_ramp_time) if cfg.walk_gait_reference == "cpg" else 0.0,
                float(self.dt),
            )
            if cfg.walk_objective_mode == "dial":
                startup_cap = max(startup_cap, float(cfg.walk_velocity_ramp_time))
            layout = interface["observation_layout"]
            layout["version"] = "h1_joint_target_qpos_qvel_task34_memory"
            layout["observation_size"] += 13
            layout["task_feature_names"] += [
                "walk_swing_seen_left", "walk_swing_seen_right",
                "walk_foot_loaded_left", "walk_foot_loaded_right",
                "walk_swing_eligible_left", "walk_swing_eligible_right",
                "walk_advance_since_landing_left", "walk_advance_since_landing_right",
                "walk_completed_steps_fraction_left", "walk_completed_steps_fraction_right",
                "walk_goal_hold_fraction", "startup_elapsed_fraction",
                "contact_force_ramp_fraction",
            ]
            interface["task"]["memory_contract"] = {
                "events": "loaded_to_unloaded_with_continuous_opposite_support; full_air_cancels_swing",
                "loaded_normal_force_gt": 0.0,
                "lift_clearance": float(cfg.walk_lift_height),
                "touchdown_clearance": float(cfg.walk_touchdown_height),
                "minimum_step_advance": float(cfg.walk_step_min_distance),
                "required_steps_per_foot": int(cfg.walk_min_steps_per_foot),
                "required_goal_hold_time": float(cfg.walk_success_hold_time),
                "minimum_body_progress": float(cfg.walk_min_body_progress),
                "minimum_support_progress": float(cfg.walk_min_support_progress),
                "maximum_goal_box_speed": float(cfg.walk_success_max_box_speed),
                "goal_tolerance": float(cfg.goal_eps),
                "reset_reference": "deterministic_robot_home",
                "landing_advance_encoding": "signed_metres_unclipped",
                "step_encoding": "completed_fraction_capped_at_required_steps",
                "hold_encoding": "elapsed_fraction_clipped_zero_one",
                "startup_elapsed_cap": startup_cap,
                "approach_time": float(cfg.approach_time),
                "force_ramp_time": float(cfg.force_ramp_time),
                "gait_ramp_time": (float(cfg.gait_ramp_time)
                                   if cfg.walk_gait_reference == "cpg" else None),
                "contact_clock": "acquired_times_clipped_contact_age_over_force_ramp_time",
            }
        if cfg.walk_objective_mode == "dial":
            interface["task"]["locomotion_objective"] = {
                "mode": cfg.walk_objective_mode,
                "target_vx": float(cfg.target_vx),
                "target_vy": 0.0,
                "velocity_ramp_time": float(cfg.walk_velocity_ramp_time),
                "height_target": float(cfg.walk_height_target),
                "weights": {
                    "gait": float(cfg.w_gait), "velocity": float(cfg.w_vel),
                    "upright": float(cfg.w_upright), "height": float(cfg.w_height),
                    "yaw": float(cfg.w_yaw), "yaw_rate": float(cfg.w_angvel),
                    "effort": float(cfg.w_walk_effort),
                },
                "velocity_frame": "torso_body_xy",
                "upright_reference": "pelvis_up",
                "yaw_reference": "wrapped_torso_yaw_to_zero",
                "yaw_rate_reference": "torso_body_z_to_zero_in_radians_per_second",
                "clock": "pre_transition_step_times_dt",
                "effort": "first_substep_applied_torque_over_actuator_upper_limits_squared",
            }
        if cfg.walk_force_startup_mode == "synchronized":
            interface["task"]["force_startup"] = {
                "mode": "synchronized",
                "execution_scale": "min(acquired_contact_age_ramp, fixed_global_benchmark_ramp)",
                "global_onset_seconds": float(cfg.approach_time),
                "ramp_seconds": float(cfg.force_ramp_time),
                "clock": "pre_transition_step_times_dt; fixed_across_physics_substeps",
                "benchmark": "unchanged_fixed_global_ramp_independent_of_action",
                "acquisition": "unchanged_finite_geometry_near_band_or_measured_force",
            }
            interface["task"]["memory_contract"]["contact_clock"] = (
                "selected_execution_scale_min_contact_and_fixed_global_ramps"
            )
        if cfg.walk_box_goal_mode == "coast":
            interface["task"]["box_goal_objective"] = {
                "mode": "friction_only_coast",
                "formula": "x + vx*abs(vx)/(2*executed_x_frictionloss/executed_box_mass)",
                "state_frame": "box world x position and velocity at the reward pipeline state",
                "assumption": "task-owned hand retraction then one-dimensional Coulomb coasting",
                "weight": float(cfg.w_box),
                "force_taper": (
                    "uses predicted stopping location; switches both hands to the task-owned "
                    "retract realization; success remains actual-state only"
                ),
                "success_and_safety": "unchanged actual-state checks; prediction is not a certificate",
            }
        return interface

    @property
    def policy_action_transform(self):
        """Task-owned residual chart used only to learn a P4 policy prior.

        The solver action remains the absolute 23D task action documented by
        :attr:`policy_interface`.  PPO, however, should explore around the
        robot's executable home pose rather than around the midpoint of every
        joint bound.  Without a trajectory reference, its chart spans the
        complete DIAL joint-target interval.  With the shared reference, zero
        already realizes the verified gait and the learned prior receives only
        local correction authority; MGA itself retains the full task chart.
        """
        if not (self._is_walk and self._bcfg.walk_leg_control == "joint_target"):
            return None
        bounds = np.asarray(
            self._whole_body_controller.planner_joint_bounds, dtype=np.float32
        )
        home = np.asarray(self._defaultN[:self._n_planner], dtype=np.float32)
        if getattr(self, "_walk_joint_reference", None) is None:
            leg_bias = np.asarray(
                self._whole_body_controller.planner_action_from_joints(home),
                dtype=np.float32,
            )
            leg_scale = (1.0 + np.abs(leg_bias)).astype(np.float32)
            center = "deterministic_robot_home"
            authority = "complete_robot_planner_joint_bounds"
        else:
            leg_bias = np.zeros((self._n_planner,), dtype=np.float32)
            leg_scale = np.full(
                (self._n_planner,),
                float(self._bcfg.policy_joint_reference_residual_scale),
                dtype=np.float32,
            )
            center = "time_indexed_dial_reference"
            authority = "local_residual_about_shared_low_level_reference"
        primitive_width = int(self.spec.total_width)
        action_bias = np.concatenate([
            np.zeros((primitive_width,), dtype=np.float32), leg_bias,
        ])
        action_scale = np.concatenate([
            np.full(
                (primitive_width,), float(getattr(self._bcfg, "policy_primitive_scale", 1.0)),
                dtype=np.float32,
            ),
            leg_scale,
        ])
        return {
            "schema_version": 2,
            "mapping": "clip(action_bias + action_scale * residual, -1, 1)",
            "center": center,
            "leg_authority": authority,
            "action_bias": action_bias.astype(float).tolist(),
            "action_scale": action_scale.astype(float).tolist(),
        }

    @property
    def manifold_constraint_size(self) -> int:
        """Action-controllable equality dimension used by ATACOM.

        Fixed stance constrains the six stiffness coordinates and force.  P3
        additionally constrains its three face logits, leaving both lateral
        and vertical contact coordinates free for correction after release.
        P4 leaves stiffness/contact selection to the whole-body
        policy and retains only the desired-force equality.
        """
        if self._is_walk:
            return 1
        return 10 if self._face_select else 7

    @property
    def inequality_constraint_size(self) -> int:
        return 3

    @property
    def reliability_feature_size(self) -> int:
        # Fixed across P1--P4 despite the different action dimensions.
        return 24

    @staticmethod
    def reliability_source_sha256():
        """Collection provenance, not evidence of OOD calibration coverage."""
        import hashlib
        from pathlib import Path

        root = Path(__file__).resolve().parents[4]
        paths = (
            "genedynamics/envs/domains/humanoid/box_push_brax.py",
            "genedynamics/envs/domains/humanoid/box_push_scene.py",
            "genedynamics/core/control/humanoid_contact.py",
            "genedynamics/core/control/bipedal_gait.py",
            "genedynamics/experiments/plugins/metrics/extractors.py",
            "genedynamics/robots/h1/profile.py",
            "genedynamics/envs/assets/unitree_h1/mjx_scene_h1_box_push.xml",
            "genedynamics/envs/assets/unitree_h1/mjx_h1_calf_hand_body.xml",
        )
        return {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                for path in paths}

    def reliability_contract(self, horizon_steps=None):
        """Task-owned feature/label and realization semantics; no physics replay.

        ``None`` identifies the saved per-transition signals.  A fitter must
        bind their complete-window length explicitly before deployment.  Domain
        mass/friction and seeds are provenance, not interface equality checks;
        a compatible interface is not a claim of calibration under that OOD.
        """
        from copy import deepcopy

        if horizon_steps is not None and (int(horizon_steps) != horizon_steps or horizon_steps < 1):
            raise ValueError("H1 reliability horizon_steps must be a positive integer")
        cfg = self._bcfg
        schema = deepcopy(H1_RELIABILITY_SCHEMA)
        if str(cfg.robot).lower() != "h1":
            # G1 keeps its pre-existing endpoint contract: H1 instrumentation
            # does not constitute new physical coverage for another robot.
            schema.update({
                "version": 1,
                "risk_definitions": [
                    "max(hand_force_gt_fmax)",
                    "max(nonhand_force_gt_0.5N_or_torso_up_lt_0_or_height_lt_0.6_reset_height)",
                    "mean_largest_ceil_H_over_5_relu_balance_residual_over_support_radius",
                    "mean_abs_hand_force_minus_time_ramped_tapered_benchmark_over_max_ftarget_1",
                ],
                "termination": "fall_priority_success_absorbing_retain_physical_risks_and_actual_completion",
            })
            schema.pop("physics_sampling")
            schema.pop("missing_physics_coverage")
            schema.pop("safe_gate")
            schema.pop("execution_scope")
            schema.pop("execution_context_version")
            schema.pop("initial_risk")
        fields = (
            "f_min", "f_max", "f_target", "support_radius", "force_int_max",
            "approach_time", "force_ramp_time", "contact_acquire_force",
            "contact_acquire_gap", "kp_force", "ki_force", "fast_force_loop",
            "fixed_force_target", "fixed_contact_target", "s_ref_diag", "s_scale",
            "d_damp", "emergency_retract_force", "emergency_backup_steps",
            "emergency_min_dwell_steps", "emergency_reference_rewind_steps",
            "arm_grav_comp",
            "arm_posture_kp", "arm_posture_kd",
            "arm_null_damping", "stance_force_ankle_gain", "stance_force_hip_gain",
            "stance_force_hip_deadband", "stance_force_reference",
            "stance_hip_bias", "stance_ankle_bias",
            "stance_com_ankle_gain", "stance_com_ankle_damping",
            "push_dist", "goal_eps", "unjam_yaw_eps", "unjam_release_clearance",
        )
        return {
            "schema": schema,
            "horizon_steps": None if horizon_steps is None else int(horizon_steps),
            "dt": float(self.dt), "timestep": float(cfg.timestep),
            "robot": self._robot_profile.model_id,
            "level": str(cfg.level),
            "action_layout": {
                "action_size": int(self.action_size),
                "primitive_width": int(self.spec.total_width),
                "use_base": bool(cfg.use_base), "stiffness_mode": cfg.stiffness_mode,
            },
            "parameters": {name: getattr(cfg, name) for name in fields},
            "walk_parameters": {
                name: getattr(cfg, name) for name in (
                    "walk_success_mode", "walk_leg_control", "walk_stop_distance",
                    "walk_approach_force_floor", "leg_grav_comp",
                    "emergency_knee_residual_levels",
                )
            } if self._is_walk else None,
            "policy_interface": self.policy_interface,
        }

    def validate_reliability_checkpoint(
        self, payload, *, horizon_steps, allow_abstaining_ood=False
    ):
        """Validate H1 bounds before constructing a controller.

        A development checkpoint can be inspected under the explicit
        ``model_based`` OOD policy, but only in abstaining mode.  It is never
        allowed to act as a learned veto until its active realization/domain
        has been independently validated and promoted.
        """
        expected = self.reliability_contract(horizon_steps)
        for name in ("feature_names", "risk_names", "state_feature_count",
                     "support_state_feature_count", "probability_risk_count",
                     "classification_probabilities"):
            if payload.get(name) != H1_RELIABILITY_SCHEMA[name]:
                raise ValueError(f"H1 reliability checkpoint {name} mismatch")
        metadata = payload.get("metadata") or {}
        if expected not in metadata.get("reliability_contracts", []):
            if (
                allow_abstaining_ood
                and not bool(metadata.get("performance_validated", False))
                and not bool(metadata.get("promotion_eligible", False))
            ):
                return
            raise ValueError(
                "H1 reliability contract mismatch: matching 24D features are insufficient; "
                "collect and fit the active realization, risk definition and horizon"
            )

    # --- normalized primitive -> physical pieces ---
    def _stiffness(self, s_raw):
        mode = self._bcfg.stiffness_mode
        if mode == "none":
            return jnp.eye(_STIFF_D, dtype=jnp.float32)
        if mode == "fixed":
            return stiffness_log_to_pd(self._s_ref, _STIFF_D)
        S_vec = self._s_ref + self._bcfg.s_scale * s_raw
        if mode == "euclid":
            from genedynamics.core.control.stiffness import svec2sym
            d = jax.nn.softplus(jnp.diagonal(svec2sym(S_vec, _STIFF_D)))
            return jnp.diag(d)
        return stiffness_log_to_pd(S_vec, _STIFF_D)        # log_spd: K = exp(S)

    def _force_cmd(self, nu_raw):
        cfg = self._bcfg
        if cfg.fixed_force_target:
            return jnp.float32(cfg.f_target)
        return cfg.f_min + 0.5 * (jnp.clip(nu_raw, -1.0, 1.0) + 1.0) * (cfg.f_max - cfg.f_min)

    def _unpack(self, action):
        cfg = self._bcfg
        pos = action[self.spec.r_slice]
        if cfg.use_base:
            v_base = cfg.v_base_scale * jnp.tanh(pos[:3]); jlog = pos[3:6]; a, b = pos[6], pos[7]
        else:
            v_base = jnp.zeros(3, jnp.float32); jlog = pos[:3]; a, b = pos[3], pos[4]
        # face selection: relaxed one-hot for H4; fixed rear face otherwise
        if self._face_select:
            jlog = jlog + jnp.asarray(cfg.unjam_face_logit_bias, jnp.float32)
            a = jnp.clip(a + cfg.unjam_contact_a_bias, -1.0, 1.0)
        w_face = (jax.nn.softmax(jlog / max(float(cfg.face_logit_temperature), 1e-6))
                  if self._face_select else jnp.array([1.0, 0.0, 0.0]))
        K_hand = self._stiffness(action[self.spec.s_slice])
        F_n = self._force_cmd(action[self.spec.nu_slice][0])
        a01 = 0.5 * (jnp.clip(a, -1.0, 1.0) + 1.0)
        b01 = 0.5 * (jnp.clip(b, -1.0, 1.0) + 1.0)
        if cfg.fixed_contact_target:
            a01 = jnp.float32(0.5)
            b01 = jnp.float32(0.5)
        return v_base, w_face, a01, b01, K_hand, F_n

    # --- contact-face geometry: soft-selected face point + outward normal ---
    # Face offsets/normals are in the box LOCAL frame and rotated by the box orientation
    # (the planar box can YAW), so the contact face follows the box as it turns (needed for
    # L3 unjamming; ~identity for the straight-push levels where yaw≈0).
    def _contact_target(self, box_pos, box_quat, w_face, a, b):
        h = self._half
        cfg = self._bcfg
        a_span = h
        if cfg.robot.lower() == "h1" and cfg.level.lower() == "unjam":
            # One rear-face coordinate locates the shared two-hand centre.
            # Reserve the existing hand spacing and edge margin together;
            # independently clipping either hand would change the chart.
            # This does not certify the existing soft side-face mixtures.
            a_span = h - abs(cfg.hand_offset) - cfg.contact_edge_margin
        ax = (2.0 * a - 1.0) * a_span
        # vertical contact at HAND-REACH height so the HAND (not the legs) meets the face: offset
        # from the box center up to hand_push_height +- a small b-band. (planar box yaws only, so
        # local z == world z; this puts p_c on the crate's UPPER face within the arm's reach.)
        bz_raw = (cfg.hand_push_height - box_pos[2]) + (2.0 * b - 1.0) * cfg.contact_band
        z_lim = jnp.maximum(h - cfg.contact_edge_margin, 0.0)
        bz = jnp.clip(bz_raw, -z_lim, z_lim)
        offs = jnp.stack([jnp.array([-h, ax, bz]),     # rear  (local -x face)
                          jnp.array([ax, h, bz]),      # left  (local +y face)
                          jnp.array([ax, -h, bz])])    # right (local -y face)
        nrm = jnp.stack([jnp.array([-1.0, 0.0, 0.0]),
                         jnp.array([0.0, 1.0, 0.0]),
                         jnp.array([0.0, -1.0, 0.0])])
        off_c = w_face @ offs
        n_c = w_face @ nrm
        p_c = box_pos + brax_math.rotate(off_c, box_quat)   # local face offset -> world
        n_c = brax_math.rotate(n_c, box_quat)
        return p_c, n_c / (jnp.linalg.norm(n_c) + 1e-9)

    # --- hand contact wrench (shared by π_low and the friction constraint) ---
    def _one_hand(self, ps, body, geom, p_c, n_c, K_hand, F_n):
        return hand_contact_wrench(
            self.sys,
            ps,
            body_id=body,
            geom_id=geom,
            target_position=p_c,
            contact_normal=n_c,
            stiffness=K_hand,
            desired_force=F_n,
            damping=self._bcfg.d_damp,
            local_push_axis=self._robot_profile.controller_defaults[
                "hand_push_axis"
            ],
        )

    @property
    def _supports_mga_execution_context(self):
        cfg = self._bcfg
        return (str(getattr(cfg, "robot", "")).lower() == "h1"
                and bool(getattr(cfg, "fast_force_loop", False))
                and getattr(self, "_acquisition_box", None) is not None)

    def _empty_mga_execution_context(self):
        return {
            "mga_execution_mode": jnp.int32(0),
            "mga_execution_request": jnp.int32(-1),
            "mga_unload_age": jnp.int32(0),
            "mga_emergency_zero_force": jnp.bool_(False),
            "mga_unload_entry_prepared": jnp.bool_(False),
            "mga_unload_geometry_valid": jnp.bool_(False),
            "mga_unload_targets": jnp.zeros((2, 3), jnp.float32),
            "mga_unload_normals": jnp.zeros((2, 3), jnp.float32),
            "mga_unload_stiffness_raw": jnp.zeros((6,), jnp.float32),
            "mga_unload_stiffness_matrix": jnp.zeros((3, 3), jnp.float32),
            "mga_unload_stiffness_valid": jnp.bool_(False),
        }

    @staticmethod
    def _mga_inspection_mode(info):
        """Pending mode describes this interval; otherwise inspect committed mode."""
        if info is None:
            return jnp.int32(0)
        request = jnp.asarray(info.get("mga_execution_request", -1), jnp.int32)
        committed = jnp.asarray(info.get("mga_execution_mode", 0), jnp.int32)
        return jnp.where(request >= 0, request, committed)

    def _mga_unload_entry(self, state):
        """Seal actual finite sphere/box geometry and actually applied stiffness.

        A two-centimetre target retreat is not a claim that the hands have
        already separated. Corners, an interior sphere centre, or insufficient
        bounded retreat make the entry invalid, not a guessed rear-face hold.
        """
        ps = state.pipeline_state
        hands = jnp.asarray([self._rhand_geom, self._lhand_geom])
        positions = ps.geom_xpos[hands]
        centre = ps.geom_xpos[self._box_geom]
        rotation = ps.geom_xmat[self._box_geom]
        half = self.sys.geom_size[self._box_geom]
        radii = self.sys.geom_size[hands, 0]
        local = (positions - centre) @ rotation
        excess = jnp.abs(local) - half
        axis = jnp.argmax(excess, axis=1)
        axis_mask = jax.nn.one_hot(axis, 3, dtype=jnp.float32)
        signs = jnp.take_along_axis(local, axis[:, None], axis=1)
        normals = (axis_mask * jnp.sign(signs)) @ rotation.T
        norm = jnp.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / jnp.where(norm > 0.0, norm, 1.0)
        retreat = jnp.float32(self._bcfg.contact_acquire_gap)
        targets = positions + retreat * normals

        def sphere_gaps(points):
            delta = jnp.abs((points - centre) @ rotation) - half
            return (jnp.linalg.norm(jax.nn.relu(delta), axis=1)
                    + jnp.minimum(jnp.max(delta, axis=1), 0.0) - radii)

        gaps = sphere_gaps(positions)
        target_gaps = sphere_gaps(targets)
        geometry_valid = (
            jnp.all(jnp.isfinite(positions)) & jnp.all(jnp.isfinite(rotation))
            & jnp.all(jnp.isfinite(half)) & jnp.all(half > 0.0)
            & jnp.all(jnp.isfinite(radii)) & jnp.all(radii > 0.0)
            & jnp.isfinite(retreat) & (retreat > 0.0)
            & jnp.all(jnp.sum(excess > 0.0, axis=1) == 1)
            & jnp.all(jnp.where(axis_mask > 0.0, True, jnp.abs(local) < half))
            & jnp.all(gaps > -retreat) & jnp.all(target_gaps > 0.0)
            & jnp.all(jnp.isfinite(targets)) & jnp.all(jnp.isfinite(normals))
        )
        # Cold reset has prev_action=0, hence the existing nominal stiffness.
        raw = jnp.asarray(
            state.info.get(
                "prev_realized_stiffness_raw",
                state.info["prev_action"][self.spec.s_slice],
            ),
            jnp.float32,
        )
        raw_valid = jnp.all(jnp.isfinite(raw)) & jnp.all(jnp.abs(raw) <= 1.0)
        # This finite placeholder is never executable when raw_valid is false.
        # It is not clipping an invalid command into a certified candidate.
        matrix = self._stiffness(jnp.where(raw_valid, raw, jnp.zeros_like(raw)))
        stiffness_valid = (raw_valid & jnp.all(jnp.isfinite(matrix))
                           & jnp.all(jnp.linalg.eigvalsh(matrix) > 0.0))
        return {
            "mga_unload_entry_prepared": jnp.bool_(True),
            "mga_unload_geometry_valid": geometry_valid,
            "mga_unload_targets": targets,
            "mga_unload_normals": normals,
            "mga_unload_stiffness_raw": raw,
            "mga_unload_stiffness_matrix": matrix,
            "mga_unload_stiffness_valid": stiffness_valid,
        }

    def _prepare_mga_execution_state(self, state, mode):
        """Pure preparation; never mutate the measured state or commit a mode."""
        mode = jnp.asarray(mode, jnp.int32)
        info = {**self._empty_mga_execution_context(), **state.info}
        committed = info["mga_execution_mode"]
        pending = info["mga_execution_request"]
        prepared = info["mga_unload_entry_prepared"]
        terminal = jnp.asarray(info.get("task_success", 0.0)) > 0.5
        # A pending/committed UNLOAD without a seal is malformed. In particular
        # the nominal model must not repair it by recapturing different geometry.
        capture = ((mode == 1) & (committed == 0) & (pending != 1) & ~prepared & ~terminal)
        entry_keys = tuple(self._mga_unload_entry_keys())
        sealed = {key: info[key] for key in entry_keys}
        sealed = jax.lax.cond(
            capture, lambda _: self._mga_unload_entry(state),
            lambda _: sealed, operand=None,
        )
        info.update(sealed)
        cancel_uncommitted = (mode == 0) & (committed == 0) & ~terminal
        for key in ("mga_unload_entry_prepared", "mga_unload_geometry_valid",
                    "mga_unload_stiffness_valid"):
            info[key] = jnp.where(cancel_uncommitted, False, info[key])
        ready = self._mga_unload_info_is_ready(info)
        # Mode 1 prefers a geometry-sealed retract.  At a corner or after a
        # contact switch there may be no honest single-face normal to seal;
        # that interval uses the task-owned zero-Cartesian-force branch.
        # Fixed-stance tasks can retract along a sealed contact normal.  During
        # locomotion, a switching/moving contact can make that world-frame
        # position spring inject an impact even when its geometric direction
        # is valid.  P4 instead uses the Cartesian zero-wrench branch below,
        # together with null-space joint damping and gravity compensation.
        # Invalid geometry also uses that direction-free branch.
        zero_force = (mode == 1) & (
            ~ready | jnp.asarray(
                getattr(self, "_walk_requires_locomotion", False)
            )
        )
        info["mga_emergency_zero_force"] = jnp.where(
            terminal, info["mga_emergency_zero_force"], zero_force
        )
        # An absorbing task cannot open/cancel an execution session.
        info["mga_execution_request"] = jnp.where(terminal, pending, mode)
        return state.replace(info=info)

    @staticmethod
    def _mga_unload_entry_keys():
        return (
            "mga_unload_entry_prepared", "mga_unload_geometry_valid",
            "mga_unload_targets", "mga_unload_normals",
            "mga_unload_stiffness_raw", "mga_unload_stiffness_matrix",
            "mga_unload_stiffness_valid",
        )

    def _mga_unload_entry_is_ready(self, state):
        return self._mga_unload_info_is_ready(state.info)

    @staticmethod
    def _mga_unload_info_is_ready(info):
        return (jnp.asarray(info.get("mga_unload_entry_prepared", False))
                & jnp.asarray(info.get("mga_unload_geometry_valid", False))
                & jnp.asarray(info.get("mga_unload_stiffness_valid", False))
                & jnp.all(jnp.isfinite(info["mga_unload_targets"]))
                & jnp.all(jnp.isfinite(info["mga_unload_normals"]))
                & jnp.all(jnp.isfinite(info["mga_unload_stiffness_matrix"]))
                & jnp.all(jnp.isfinite(info["mga_unload_stiffness_raw"]))
                & jnp.all(jnp.abs(info["mga_unload_stiffness_raw"]) <= 1.0)
                & jnp.all(jnp.linalg.eigvalsh(info["mga_unload_stiffness_matrix"]) > 0.0))

    @property
    def mga_execution_context(self):
        """Host execution plus pure preparation; other robots retain legacy API."""
        if not self._supports_mga_execution_context:
            return None
        step_jit = jax.jit(self.step)
        score_emergency_transition_jit = jax.jit(
            self._score_emergency_transition
        )

        def execute(state, action, mode):
            if any(isinstance(leaf, jax.core.Tracer)
                   for leaf in jax.tree_util.tree_leaves((state, action, mode))):
                raise TypeError("MGA execution callback must run on the host")
            selected = int(mode)
            if selected not in (0, 1):
                raise ValueError("MGA execution mode must be NORMAL(0) or UNLOAD(1)")
            prepared = self._prepare_mga_execution_state(state, selected)
            # A valid but uncertified mitigation is allowed: it is not a
            # certified-safe action and its real samples must remain visible.
            # An invalid retract entry executes zero Cartesian arm force; it
            # is never repaired into a guessed contact normal.
            return step_jit(prepared, action)

        def score_emergency(state, actions, aug_lambda, aug_rho):
            """Score the UNLOAD prefix that will precede NORMAL recovery.

            This is intentionally a host capability.  Nesting ``self.step``
            inside a second H1 certificate JIT duplicates the complete MJX
            transition executable and exceeds memory-limited CPU containers.
            The two functional transitions are still model based: they advance
            a prepared copy of the measured state, never the real environment.
            P4 normally uses the configured receding viability prefix. An
            explicit minimum dwell changes the hybrid suffix actually being
            deployed: certify exactly the remaining UNLOAD intervals, after
            which the backend separately certifies the complete NORMAL
            recovery horizon. Requiring additional fictitious UNLOAD steps
            there can reject a safe dwell-to-recovery transition.
            """
            if any(isinstance(leaf, jax.core.Tracer)
                   for leaf in jax.tree_util.tree_leaves(
                       (state, actions, aug_lambda, aug_rho)
                   )):
                raise TypeError("MGA emergency host certificate cannot be traced")
            current = state
            scores, risks = [], []
            dense = jnp.asarray(actions)
            backup_steps = (
                int(self._bcfg.emergency_backup_steps)
                if self._walk_requires_locomotion else 2
            )
            if (
                self._walk_requires_locomotion
                and self._bcfg.emergency_min_dwell_steps > 1
            ):
                committed_age = int(np.asarray(jax.device_get(
                    state.info.get("mga_unload_age", 0)
                )))
                backup_steps = max(
                    1,
                    int(self._bcfg.emergency_min_dwell_steps) - committed_age,
                )
            for index in range(min(backup_steps, int(dense.shape[0]))):
                prepared = self._prepare_mga_execution_state(
                    current, jnp.int32(1)
                )
                action = dense[index]
                predicted = step_jit(prepared, action)
                score, risk = score_emergency_transition_jit(
                    prepared, predicted, action,
                    jnp.asarray(aug_lambda, jnp.float32),
                    jnp.asarray(aug_rho, jnp.float32),
                )
                scores.append(score)
                risks.append(risk)
                current = predicted
            if not scores:
                raise ValueError("emergency certificate requires a nonempty action sequence")
            return (
                jnp.mean(jnp.stack(scores)),
                self._aggregate_sequence_risks(jnp.stack(risks)),
            )

        context = {
            "schema_version": 1, "normal_mode": 0, "emergency_mode": 1,
            "prepare_state": self._prepare_mga_execution_state,
            "mode_from_state": lambda state: jnp.asarray(
                state.info.get("mga_execution_mode", 0), jnp.int32),
            "normal_recovery_ready": self.normal_recovery_ready,
            "step": execute,
            "score_emergency_host": score_emergency,
        }
        return context

    def normal_recovery_ready(self, state):
        """Whether the committed UNLOAD dwell permits a NORMAL exit."""
        return jnp.asarray(
            state.info.get("mga_unload_age", 0), jnp.int32
        ) >= jnp.int32(self._bcfg.emergency_min_dwell_steps)

    def _mga_interval_state(self, state):
        # An ordinary env.step (including a raw policy's rollout) defaults to
        # NORMAL; only an explicit request can execute UNLOAD. This differs
        # deliberately from post-state inspection of the committed mode.
        request = jnp.asarray(state.info.get("mga_execution_request", -1), jnp.int32)
        mode = jnp.where(request >= 0, request, 0)
        prepared = self._prepare_mga_execution_state(state, mode)
        info = dict(prepared.info)
        terminal = jnp.asarray(info.get("task_success", 0.0)) > 0.5
        info["force_int"] = jnp.where(
            (mode == 1) & ~terminal, jnp.minimum(info["force_int"], 0.0), info["force_int"])
        return prepared.replace(info=info)

    def _mga_finish_context(self, previous_info, info, terminal):
        mode = jnp.asarray(previous_info["mga_execution_request"], jnp.int32)
        previous_mode = jnp.asarray(
            previous_info["mga_execution_mode"], jnp.int32
        )
        info["mga_execution_mode"] = jnp.where(
            terminal, previous_mode, mode)
        info["mga_execution_request"] = jnp.int32(-1)
        previous_age = jnp.asarray(
            previous_info.get("mga_unload_age", 0), jnp.int32
        )
        committed_age = jnp.where(
            mode == 1,
            jnp.where(previous_mode == 1, previous_age + 1, jnp.int32(1)),
            jnp.int32(0),
        )
        info["mga_unload_age"] = jnp.where(
            terminal, previous_age, committed_age
        )
        clear = (~terminal) & (mode == 0)
        for key in ("mga_unload_entry_prepared", "mga_unload_geometry_valid",
                    "mga_unload_stiffness_valid"):
            info[key] = jnp.where(clear, False, info[key])
        return info

    def _unload_hand_contact(self, ps, action, info, ordinary):
        """Actual sealed Cartesian realization, not a force-coordinate sentinel."""
        stiffness = info["mga_unload_stiffness_matrix"]
        measured = self._box_contact_force(ps)
        integral = jnp.minimum(jnp.asarray(info["force_int"], jnp.float32), 0.0)
        effective = jnp.clip(
            -self._bcfg.kp_force * measured + integral, -self._bcfg.f_max, 0.0)
        targets, normals = info["mga_unload_targets"], info["mga_unload_normals"]
        right = self._one_hand(ps, self._rhand_body, self._rhand_geom,
                               targets[0], normals[0], stiffness, 0.5 * effective)
        left = self._one_hand(ps, self._lhand_body, self._lhand_geom,
                              targets[1], normals[1], stiffness, 0.5 * effective)
        radii = self.sys.geom_size[jnp.asarray([self._rhand_geom, self._lhand_geom]), 0]
        result = {
            **ordinary, **right, "left": left, "n_c": normals[0],
            "p_surface": jnp.mean(targets - radii[:, None] * normals, axis=0),
            "F_n": jnp.float32(0.0), "F_n_cmd": jnp.float32(0.0),
            "F_eff": effective, "force_scale": jnp.float32(0.0),
            "approach_alpha": jnp.float32(1.0),
            # Keep the normal/UNLOAD pytrees identical. Historical H1 walk
            # has no measured-support hook; fixed/strict P4 already own it.
            **({"support_load": jnp.clip(measured, 0.0, self._bcfg.f_target)}
               if "support_load" in ordinary else {}),
        }
        if "arm_task_scale" in ordinary:
            result["arm_task_scale"] = jnp.float32(1.0)
        if "arm_control_scale" in ordinary:
            result["arm_control_scale"] = jnp.float32(1.0)
        return result

    def _zero_force_hand_contact(self, ps, action, info, ordinary):
        """Stop task-space pushing for one receding emergency interval.

        A Cartesian velocity damper or a posture/gravity hold can both inject
        an impact when switched at an already compressed rigid contact.  P4's
        two-step recursive certificate therefore evaluates a zero-wrench arm
        release while keeping the shared DIAL gait reference continuous.  The
        measured successor is replanned after the first 20 ms interval.
        """
        del action

        continuing = jnp.asarray(
            info.get("mga_execution_mode", 0), jnp.int32
        ) == 1
        retract_each = 0.5 * jnp.float32(
            self._bcfg.emergency_retract_force
        )
        retract_wrench = jnp.asarray([-retract_each, 0.0, 0.0])

        def zero_hand(hand):
            result = dict(hand)
            result["wrench"] = jnp.where(
                continuing, retract_wrench, jnp.zeros_like(hand["wrench"])
            )
            if "p_c" in result and "p_hand" in result:
                result["p_c"] = result["p_hand"]
            if "f_n" in result:
                result["f_n"] = jnp.float32(0.0)
            if "f_t" in result:
                result["f_t"] = result["wrench"]
            return result

        right = zero_hand(ordinary)
        left = zero_hand(ordinary["left"])
        result = {**ordinary, **right, "left": left}
        if "p_surface" in result and "p_hand" in right and "p_hand" in left:
            result["p_surface"] = 0.5 * (right["p_hand"] + left["p_hand"])
        for key in ("F_n", "F_n_cmd", "F_eff", "force_scale",
                    "support_load"):
            if key in result:
                result[key] = jnp.float32(0.0)
        if "arm_task_scale" in result:
            result["arm_task_scale"] = jnp.float32(1.0)
        if "arm_control_scale" in result:
            result["arm_control_scale"] = jnp.float32(1.0)
        if "arm_posture_position_scale" in result:
            result["arm_posture_position_scale"] = jnp.float32(0.0)
        if "approach_alpha" in result:
            result["approach_alpha"] = jnp.float32(1.0)
        return result

    def _terminal_coast_hand_contact(self, ps, action, info, ordinary):
        """Actively detach the hands once the P4 coast condition is reached.

        Zeroing the requested normal force is not a physical release: a
        walking robot can keep dragging the box through passive arm contact.
        Reuse the task-owned retract realization so rollout scoring and the
        executed transition see the same geometric separation.  This is a
        terminal task mode, independent of whether MGA requested EMERGENCY.
        """
        retract_info = {
            **info,
            "mga_execution_mode": jnp.int32(1),
        }
        return self._zero_force_hand_contact(
            ps, action, retract_info, ordinary
        )

    def realized_hand_stiffness(self, state, action):
        """Match actual post-state realization, including a consumed UNLOAD request."""
        nominal = self._unpack(action)[4]
        if not self._supports_mga_execution_context:
            return nominal
        return jnp.where(
            self._mga_inspection_mode(state.info) == 1,
            state.info.get("mga_unload_stiffness_matrix", nominal), nominal)

    def _hand_contact(self, ps, action, info=None):
        ordinary = self._normal_hand_contact(ps, action, info)
        if not self._supports_mga_execution_context or info is None:
            return ordinary
        context_info = {**self._empty_mga_execution_context(),
                        "force_int": jnp.float32(0.0), **info}
        mode = self._mga_inspection_mode(info)
        terminal_coast = (
            bool(getattr(self, "_walk_requires_locomotion", False))
            and getattr(self._bcfg, "walk_box_goal_mode", "position") == "coast"
            and self.walk_force_scale(ps, info) <= 0.0
        )
        # Metric extraction visits already-executed, host-concrete states one
        # at a time.  Building a fresh lax.cond for every such state makes JAX
        # compile and retain a new conditional executable on every visit.  A
        # concrete mode can be selected on the host without changing either
        # branch; traced rollout/control paths must retain the dynamic cond.
        if not isinstance(mode, jax.core.Tracer) and not isinstance(
            terminal_coast, jax.core.Tracer
        ):
            if bool(np.asarray(terminal_coast)):
                return self._terminal_coast_hand_contact(
                    ps, action, context_info, ordinary
                )
            if int(np.asarray(mode)) != 1:
                return ordinary
            return (
                self._unload_hand_contact(ps, action, context_info, ordinary)
                if (
                    bool(self._mga_unload_info_is_ready(context_info))
                    and not bool(context_info["mga_emergency_zero_force"])
                )
                else self._zero_force_hand_contact(
                    ps, action, context_info, ordinary
                )
            )
        return jax.lax.cond(
            terminal_coast,
            lambda _: self._terminal_coast_hand_contact(
                ps, action, context_info, ordinary
            ),
            lambda _: jax.lax.cond(
                mode == 1,
                lambda __: jax.lax.cond(
                    self._mga_unload_info_is_ready(context_info)
                    & ~context_info["mga_emergency_zero_force"],
                    lambda ___: self._unload_hand_contact(
                        ps, action, context_info, ordinary
                    ),
                    lambda ___: self._zero_force_hand_contact(
                        ps, action, context_info, ordinary
                    ),
                    operand=None,
                ),
                lambda __: ordinary,
                operand=None,
            ),
            operand=None,
        )

    def _normal_hand_contact(self, ps, action, info=None):
        # TWO-hand push: both hands run the Cartesian contact impedance toward symmetric points on
        # the selected face (right -> -y, left -> +y in the box frame). Top-level keys = the RIGHT
        # hand (the manifold / metrics read these); the 'left' dict drives the left arm in _control.
        v_base, w_face, a, b, K_hand, F_n_cmd = self._unpack(action)
        # The task owns the execution clock; this does not alter the separate
        # position-impedance approach or claim that zero feed-forward is zero contact force.
        force_scale = self._normal_force_startup_scale(info)
        F_n = F_n_cmd * force_scale
        if self._walk_requires_locomotion:
            F_n = F_n * self.walk_force_scale(ps, info)
        measured = self._box_contact_force(ps)
        force_int = (jnp.float32(0.0) if info is None
                     else jnp.asarray(info.get("force_int", 0.0), jnp.float32))
        if self._walk_requires_locomotion:
            force_int = jnp.minimum(
                force_int, self._bcfg.force_int_max * self.walk_force_scale(ps, info)
            )
        # F_n is the TOTAL desired two-hand force.  The old implementation applied it to EACH
        # hand (2x feed-forward) and held that wrench open-loop for all physics substeps.
        F_eff = jnp.clip(
            F_n + self._bcfg.kp_force * (F_n - measured) + force_int,
            -self._bcfg.f_max,
            self._bcfg.f_max,
        )
        box_pos = ps.x.pos[self._box_idx - 1]
        box_quat = ps.x.rot[self._box_idx - 1]
        p_surface, n_c = self._contact_target(box_pos, box_quat, w_face, a, b)
        if self._walk_requires_locomotion and info is not None and "box_goal_x" in info:
            # Stop advancing the normal contact target beyond the goal face.
            # This prevents the arms from chasing an overshooting box while
            # the feet are still catching up to satisfy locomotion success.
            p_surface = p_surface.at[0].set(jnp.minimum(
                p_surface[0], info["box_goal_x"] - self._half
            ))
        # The semantic target lies on the box surface, while the controlled
        # point is the center of the physical collision geom.  Offset its
        # target outward so zero position residual means surface contact, not
        # penetration by one hand radius/half-extent.
        p_c = p_surface + float(
            self._robot_profile.controller_defaults["hand_forward_extent"]
        ) * n_c
        off = self._bcfg.hand_offset * brax_math.rotate(jnp.array([0.0, 1.0, 0.0]), box_quat)
        if info is None:
            approach_alpha = jnp.float32(1.0)
        else:
            approach_alpha = jnp.clip(
                jnp.asarray(info.get("step", 0), jnp.float32) * self.dt
                / max(float(self._bcfg.approach_time), 1e-6),
                0.0,
                1.0,
            )
        p_right = self._rhand_home + approach_alpha * (p_c - off - self._rhand_home)
        p_left = self._lhand_home + approach_alpha * (p_c + off - self._lhand_home)
        F_each = 0.5 * F_eff
        right = self._one_hand(ps, self._rhand_body, self._rhand_geom, p_right, n_c, K_hand, F_each)
        left = self._one_hand(ps, self._lhand_body, self._lhand_geom, p_left, n_c, K_hand, F_each)
        arm_task_scale = self.walk_arm_task_scale(ps, info)
        contact = dict(v_base=v_base, n_c=n_c, p_surface=p_surface,
                       F_n=F_n, F_n_cmd=F_n_cmd,
                       F_eff=F_eff, force_scale=force_scale,
                       approach_alpha=approach_alpha,
                       arm_task_scale=arm_task_scale,
                       arm_control_scale=jnp.float32(1.0),
                       arm_posture_position_scale=arm_task_scale,
                       left=left, **right)
        if not self._is_walk:
            # At low fixed-stance loads, bracing against a force that has not
            # yet been realized drives the hands into the box and amplifies
            # the contact impulse.  At larger loads, waiting for measurement
            # alone can lose the support margin after a brief contact drop.
            # Reuse the controller's existing load deadband as a smooth
            # anticipation scale selected by the task's nominal load:
            # measured at/below the deadband, commanded by twice the deadband,
            # and blended between.  Using the nominal load (rather than the
            # instantaneous ramp value) gives a 30 N step its validated brace
            # from ramp onset without reintroducing the 15 N pre-contact kick.
            realized = jnp.minimum(F_n, measured)
            deadband = max(float(self._bcfg.stance_force_hip_deadband), 1.0e-6)
            anticipation = jnp.clip(
                (jnp.float32(self._bcfg.f_target) - deadband) / deadband,
                0.0, 1.0,
            )
            contact["support_load"] = realized + anticipation * (F_n - realized)
        elif self._walk_requires_locomotion:
            # A requested push is not an external load once the hand loses
            # contact during locomotion.  Scope measured support allocation to
            # strict P4: fixed-stance P1--P3 retain their validated commanded-
            # load schedule above, while historical non-locomotion P4 remains
            # unchanged.
            contact["support_load"] = jnp.minimum(F_n, measured)
        if self._walk_requires_locomotion:
            if self._bcfg.walk_leg_control in {
                    "support_phase", "support_phase_foot_level", "joint_target"}:
                # Planned swing is not evidence that the foot has unloaded.
                # Use measured support for both nominal-gait capture and the
                # load feed-forward applied around direct DIAL joint targets.
                # At zero hand load this does not alter the DIAL leg law.
                contact.update(self._support_feedback(ps, self._foot_contact_loads(ps)))
            if self._bcfg.walk_leg_control in {"support_phase", "support_phase_foot_level"}:
                if self._bcfg.walk_leg_control == "support_phase_foot_level":
                    # Experimental nominal-foot levelling is not part of the
                    # support_phase candidate: phase-only switching can still
                    # jump a loaded ankle's reference and is not promoted.
                    phases = self._walk_phases(info)
                    contact["swing_foot_pitch_target"] = -brax_math.quat_to_euler(
                        ps.x.rot[self._pelvis_idx - 1]
                    )[1]
                    contact["swing_feet"] = phases < self._bcfg.gait_swing_frac
        return contact

    def _walk_phases(self, info):
        step = 0 if info is None else info.get("step", 0)
        return (
            jnp.asarray(step, jnp.float32) * self.dt * self._bcfg.gait_cadence
            + jnp.array([0.0, 0.5])
        ) % 1.0

    def _feet_on_ground(self, ps):
        """Actual foot-floor contact, independent of the desired swing phase."""
        contacts = ps.contact
        on_floor = jnp.any(contacts.geom == self._floor_geom, axis=1)
        bodies = jnp.asarray(self.sys.geom_bodyid)[contacts.geom]
        return jax.vmap(lambda foot: jnp.any(
            on_floor & jnp.any(bodies == foot, axis=1) & (contacts.dist <= 0.0)
        ))(self._foot_body_ids)

    def _foot_contact_loads(self, ps):
        """Compressive foot-floor loads, for both native-converted and MJX states."""
        contacts = ps.contact
        on_floor = jnp.any(contacts.geom == self._floor_geom, axis=1)
        bodies = jnp.asarray(self.sys.geom_bodyid)[contacts.geom]
        normal = jnp.maximum(jnp.array([
            _mjx_support.contact_force(self.sys, ps, i)[0]
            for i in range(contacts.dist.shape[0])
        ]), 0.0)
        return jax.vmap(lambda foot: jnp.sum(jnp.where(
            on_floor & jnp.any(bodies == foot, axis=1) & (contacts.dist <= 0.0),
            normal, 0.0,
        )))(self._foot_body_ids)

    def _walk_contact_diagnostics(self, ps):
        """Raw physical signals, not a static balance or safety certificate.

        Capsule--plane distances are available for separated feet in MJX.
        A geometric floor contact can carry zero load; expose both quantities
        separately rather than equating nominal gait phase with support.
        """
        contacts = ps.contact
        on_floor = jnp.any(contacts.geom == self._floor_geom, axis=1)
        bodies = jnp.asarray(self.sys.geom_bodyid)[contacts.geom]
        valid = jnp.all(
            (contacts.geom >= 0) & (contacts.geom < self.sys.geom_bodyid.shape[0]),
            axis=1,
        )
        clearance = jax.vmap(lambda foot: jnp.min(jnp.where(
            valid & on_floor & jnp.any(bodies == foot, axis=1),
            contacts.dist, jnp.inf,
        ), initial=jnp.inf))(self._foot_body_ids)
        torso_up = brax_math.rotate(
            jnp.array([0.0, 0.0, 1.0]), ps.x.rot[self._torso_idx - 1]
        )[2]
        return {
            "foot_normal_loads": self._foot_contact_loads(ps),
            "foot_floor_clearance": clearance,
            "foot_ground_contact": clearance <= 0.0,
            "robot_subtree_com": ps.subtree_com[self._pelvis_idx],
            "torso_up": torso_up,
            "torso_height": ps.x.pos[self._torso_idx - 1, 2],
            "box_forward_velocity": ps.xd.vel[self._box_idx - 1, 0],
        }

    def _support_feedback(self, ps, foot_loads):
        """Continuous, task-owned allocation of the two-ankle design authority.

        Normal/full support preserves total weight two even on a single foot;
        below body weight it fades with actual support load, reaching zero in
        flight.  The common capture/balance reference fades to the pelvis in
        the same limit, rather than jumping when the last contact disappears.
        """
        loads = jnp.maximum(foot_loads, 0.0)
        total = jnp.sum(loads)
        denominator = jnp.maximum(total, max(self._robot_weight, 1e-6))
        pelvis_xy = ps.x.pos[self._pelvis_idx - 1, :2]
        feet_xy = ps.site_xpos[self._feet_site_id, :2]
        reference_xy = (
            jnp.sum(loads[:, None] * feet_xy, axis=0)
            + (denominator - total) * pelvis_xy
        ) / denominator
        return {
            "stance_support_weights": 2.0 * loads / denominator,
            "support_reference_xy": reference_xy,
        }

    def _box_contact_forces(self, ps):
        """Separate hand, wall, and invalid robot-body contact with the box."""
        c = ps.contact
        on_box = (c.geom[:, 0] == self._box_geom) | (c.geom[:, 1] == self._box_geom)
        on_hand = ((c.geom[:, 0] == self._rhand_geom) | (c.geom[:, 1] == self._rhand_geom)
                   | (c.geom[:, 0] == self._lhand_geom) | (c.geom[:, 1] == self._lhand_geom))
        on_wall = jnp.any(
            (c.geom[:, :, None] == self._wall_geoms[None, None, :]), axis=(1, 2)
        )
        on_floor = ((c.geom[:, 0] == self._floor_geom) | (c.geom[:, 1] == self._floor_geom))
        n = c.dist.shape[0]
        fn = jnp.array([_mjx_support.contact_force(self.sys, ps, i)[0] for i in range(n)])
        active = on_box & (c.dist < 0)
        right_pair = (
            ((c.geom[:, 0] == self._box_geom) & (c.geom[:, 1] == self._rhand_geom))
            | ((c.geom[:, 1] == self._box_geom) & (c.geom[:, 0] == self._rhand_geom))
        )
        return {
            "hand": jnp.sum(jnp.where(active & on_hand, jnp.abs(fn), 0.0)),
            "wall": jnp.sum(jnp.where(active & on_wall, jnp.abs(fn), 0.0)),
            "nonhand": jnp.sum(jnp.where(
                active & ~on_hand & ~on_wall & ~on_floor, jnp.abs(fn), 0.0
            )),
            # Current right-hand compression, not the two-hand acquisition
            # latch or the legacy absolute-force summaries above.  Exact
            # touching contact may carry load; separated/tensile rows cannot.
            "right_supported": jnp.any(right_pair & (c.dist <= 0.0) & (fn > 0.0)),
        }

    def _box_contact_force(self, ps):
        """Backward-compatible hand-only physical box force."""
        return self._box_contact_forces(ps)["hand"]

    def _hand_box_acquisition_gaps(self, ps):
        """Finite-box surface gaps for the supported H1 sphere contactors."""
        hands = jnp.asarray([self._rhand_geom, self._lhand_geom])
        points = (ps.geom_xpos[hands] - ps.geom_xpos[self._box_geom]) @ ps.geom_xmat[self._box_geom]
        return self._acquisition_box.jax_sdf(points) - self.sys.geom_size[hands, 0]

    def _contact_acquisition_detected(self, ps, action, info, measured):
        if self._acquisition_box is not None:
            gaps = self._hand_box_acquisition_gaps(ps)
            in_approach_band = jnp.all(
                jnp.isfinite(gaps) & (gaps >= 0.0)
                & (gaps <= self._bcfg.contact_acquire_gap)
            )
        else:
            # Preserve the existing non-sphere/G1 realization.  Its capsule
            # geometry is outside the validated H1 acquisition repair.
            contact = self._hand_contact(ps, action, info)
            gap_r = jnp.dot(contact["p_hand"] - contact["p_c"], contact["n_c"])
            gap_l = jnp.dot(contact["left"]["p_hand"] - contact["left"]["p_c"], contact["n_c"])
            in_approach_band = jnp.maximum(gap_r, gap_l) <= self._bcfg.contact_acquire_gap
        return (measured >= self._bcfg.contact_acquire_force) | in_approach_band

    # --- pi_low: task contact command -> robot-specific whole-body torque ---
    def _control(self, ps, action, info):
        contact = self._hand_contact(ps, action, info)
        return self._whole_body_controller.torque(
            ps, contact, action, info, self.spec.total_width
        )

    # --- reset ---
    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)
        box_x0 = state.pipeline_state.x.pos[self._box_idx - 1, 0]
        info = dict(state.info)
        info["box_goal_x"] = box_x0 + self._bcfg.push_dist
        info["box_x0"] = box_x0
        info["step"] = jnp.int32(0)
        info["contact_acquired"] = jnp.float32(0.0)
        info["contact_step"] = jnp.int32(-1)
        info["force_int"] = jnp.float32(0.0)
        info["task_success"] = jnp.float32(0.0)
        info["task_fallen"] = jnp.float32(0.0)
        info["success_padding"] = jnp.bool_(False)
        info["unjam_released"] = jnp.float32(0.0)
        info["prev_action"] = jnp.zeros((self.action_size,), jnp.float32)
        info["prev_realized_stiffness_raw"] = info["prev_action"][self.spec.s_slice]
        if self._supports_mga_execution_context:
            info.update(self._empty_mga_execution_context())
        if str(self._bcfg.robot).lower() == "h1":
            info.update(self._empty_physics_safety_samples())
        if self._walk_requires_locomotion:
            feet = state.pipeline_state.site_xpos[self._feet_site_id]
            info.update({
                "walk_body_x0": state.pipeline_state.x.pos[self._pelvis_idx - 1, 0],
                "walk_support_x0": feet[:, 0].mean(),
                "walk_foot_z0": feet[:, 2],
                "walk_landing_x": feet[:, 0],
                "walk_swing_seen": jnp.zeros((2,), jnp.bool_),
                "walk_forward_steps": jnp.zeros((2,), jnp.int32),
                "walk_body_progress": jnp.float32(0.0),
                "walk_support_progress": jnp.float32(0.0),
                "walk_progress_potential": jnp.float32(0.0),
                "walk_progress_delta": jnp.float32(0.0),
                "walk_new_steps": jnp.float32(0.0),
                "walk_goal_ready": jnp.bool_(False),
                "walk_goal_hold_time": jnp.float32(0.0),
            })
            if (self._walk_joint_reference is not None
                    and self._bcfg.emergency_reference_rewind_steps > 0):
                info["walk_reference_step"] = jnp.int32(0)
                info["walk_reference_recovery_anchor"] = jnp.int32(0)
                info["walk_reference_phase_error"] = jnp.float32(0.0)
            if self._bcfg.walk_leg_control == "joint_target":
                info["walk_foot_loaded"] = self._foot_contact_loads(state.pipeline_state) > 0.0
                info["walk_swing_eligible"] = jnp.zeros((2,), jnp.bool_)
        # The base reset builds its observation before this task installs the
        # goal.  Return the actual goal distance to the first policy/prior call.
        return state.replace(info=info, obs=self._get_obs(state.pipeline_state, info))

    def walk_force_scale(self, ps, info):
        """Task-owned approach-to-line taper, shared by execution and scoring."""
        if not self._walk_requires_locomotion or info is None or "box_goal_x" not in info:
            return jnp.float32(1.0)
        cfg = self._bcfg
        box_reference = ps.x.pos[self._box_idx - 1, 0]
        if cfg.walk_box_goal_mode == "coast":
            # Release early enough for the executed box inertia and slide
            # resistance to stop at the line.  Using position alone keeps
            # applying force until the box crosses the line, after which a
            # walking robot can only collide with the already-fast box.
            box_reference = self._walk_coasting_box_x(ps)
        remaining = info["box_goal_x"] - box_reference
        taper = jnp.clip(
            (remaining - cfg.goal_eps)
            / max(float(cfg.walk_stop_distance - cfg.goal_eps), 1e-6), 0.0, 1.0
        )
        approaching = cfg.walk_approach_force_floor + (1.0 - cfg.walk_approach_force_floor) * taper
        return jnp.where(remaining > cfg.goal_eps, approaching, jnp.float32(0.0))

    def walk_arm_task_scale(self, ps, info):
        """Keep P4 Cartesian tracking on the current box face.

        ``_normal_hand_contact`` recomputes the target from the measured box
        pose at every transition; it is not a fixed world anchor.  Keep full
        face tracking while the coast policy still requests any push, rather
        than weakening it continuously with the force taper and letting the
        crate outrun the hands.  Once the predicted release condition makes
        that request exactly zero, release both the Cartesian task and its
        push-posture position term so the measured box can coast without a
        hidden impedance push.  Arm damping and gravity compensation remain.
        """
        if not (
            self._walk_requires_locomotion
            and self._bcfg.walk_box_goal_mode == "coast"
        ):
            return jnp.float32(1.0)
        return jnp.where(
            self.walk_force_scale(ps, info) > 0.0,
            jnp.float32(1.0),
            jnp.float32(0.0),
        )

    def _global_force_startup_scale(self, info):
        """Fixed benchmark clock, independent of acquisition and the force action."""
        cfg = self._bcfg
        elapsed = jnp.asarray(info["step"], jnp.float32) * self.dt
        return jnp.clip(
            (elapsed - cfg.approach_time) / max(float(cfg.force_ramp_time), 1e-6),
            0.0, 1.0,
        )

    def _normal_force_startup_scale(self, info):
        """NORMAL feed-forward ramp; no-info calls retain the legacy static query."""
        if info is None:
            return jnp.float32(1.0)
        acquired = jnp.asarray(info.get("contact_acquired", 0.0), jnp.float32)
        contact_step = jnp.asarray(info.get("contact_step", info.get("step", 0)), jnp.float32)
        elapsed = jnp.maximum(jnp.asarray(info.get("step", 0), jnp.float32) - contact_step, 0.0) * self.dt
        scale = acquired * jnp.clip(elapsed / max(float(self._bcfg.force_ramp_time), 1e-6), 0.0, 1.0)
        if self._bcfg.walk_force_startup_mode == "synchronized":
            # Minimum, not a product: coincident ramps must retain their shape.
            scale = jnp.minimum(scale, self._global_force_startup_scale(
                {"step": info.get("step", 0)}))
        return scale

    def requested_force_reference(self, ps, info):
        """Benchmark force target; independent of the planner's force action."""
        ramp = self._global_force_startup_scale(info)
        return jnp.float32(self._bcfg.f_target) * ramp * self.walk_force_scale(ps, info)

    def _box_reached(self, ps, info):
        reached = (jnp.abs(ps.x.pos[self._box_idx - 1, 0] - info["box_goal_x"])
                   <= self._bcfg.goal_eps)
        if str(self._bcfg.level).lower() == "unjam":
            yaw = brax_math.quat_to_euler(ps.x.rot[self._box_idx - 1])[2]
            reached = reached & (jnp.abs(yaw) <= self._bcfg.unjam_yaw_eps)
        return reached

    def _walk_progress(self, ps, info):
        """Count forward landings, excluding stance sliding and in-place lifts."""
        cfg = self._bcfg
        feet = ps.site_xpos[self._feet_site_id]
        contact_history = {}
        if self._walk_requires_locomotion and cfg.walk_leg_control == "joint_target":
            physical = self._walk_contact_diagnostics(ps)
            loaded = physical["foot_normal_loads"] > 0.0
            clearance = physical["foot_floor_clearance"]
            previous_loaded = info["walk_foot_loaded"]
            released = previous_loaded & ~loaded & loaded[::-1]
            eligible = (info["walk_swing_eligible"] | released) & ~loaded & loaded[::-1]
            lifted = eligible & (clearance >= cfg.walk_lift_height)
            # No time allowance: full flight cancels the supported exchange.
            # Landing and opposite-foot release in the same control frame is
            # still a supported transfer, not a required double-stance dwell.
            swing_seen = (info["walk_swing_seen"] | lifted) & jnp.any(loaded)
            recontact = ~previous_loaded & loaded
            grounded_recontact = recontact & (clearance <= cfg.walk_touchdown_height)
            touchdown = grounded_recontact & swing_seen
            # Every physical recontact refreshes the placement anchor.  A
            # loaded slide never does, and an aborted flight never adds a step.
            landing_x = jnp.where(grounded_recontact, feet[:, 0], info["walk_landing_x"])
            swing_seen = swing_seen & ~recontact
            support_progress = landing_x.mean() - info["walk_support_x0"]
            contact_history = {
                "walk_foot_loaded": loaded,
                "walk_swing_eligible": eligible,
            }
        else:
            # Preserve the historical site-based contract outside the new
            # joint-target locomotion branch.
            height = feet[:, 2] - info["walk_foot_z0"]
            lifted_with_support = (
                (height >= cfg.walk_lift_height)
                & (height[::-1] <= cfg.walk_touchdown_height)
            )
            swing_seen = info["walk_swing_seen"] | lifted_with_support
            touchdown = swing_seen & (height <= cfg.walk_touchdown_height)
            landing_x = jnp.where(touchdown, feet[:, 0], info["walk_landing_x"])
            swing_seen = swing_seen & ~touchdown
            support_progress = feet[:, 0].mean() - info["walk_support_x0"]
        advanced = feet[:, 0] - info["walk_landing_x"] >= cfg.walk_step_min_distance
        steps = info["walk_forward_steps"] + (touchdown & advanced).astype(jnp.int32)
        body_progress = ps.x.pos[self._pelvis_idx - 1, 0] - info["walk_body_x0"]
        # Reward only progress that the moving support can plausibly sustain.
        # A forward-falling torso used to collect dense body progress while
        # both feet stayed at the reset pose.  This bounded potential allows
        # at most one step-length of body lead and pays only its transition
        # delta, so leaning/falling cannot repeatedly earn locomotion reward.
        progress_potential = jnp.clip(
            jnp.minimum(
                body_progress,
                support_progress + jnp.float32(cfg.walk_step_min_distance),
            ),
            0.0,
            jnp.float32(cfg.push_dist),
        )
        previous_potential = jnp.asarray(
            info.get("walk_progress_potential", 0.0), jnp.float32
        )
        new_steps = jnp.sum(
            steps - jnp.asarray(info["walk_forward_steps"], jnp.int32)
        ).astype(jnp.float32)
        ready = (
            self._box_reached(ps, info)
            & (body_progress >= cfg.walk_min_body_progress)
            & (support_progress >= cfg.walk_min_support_progress)
            & jnp.all(steps >= cfg.walk_min_steps_per_foot)
            & (jnp.abs(ps.xd.vel[self._box_idx - 1, 0]) <= cfg.walk_success_max_box_speed)
            & (brax_math.rotate(
                jnp.array([0.0, 0.0, 1.0]), ps.x.rot[self._torso_idx - 1]
            )[2] >= 0.8)
            & (ps.x.pos[self._torso_idx - 1, 2] >= 0.8 * self._torso_z0)
        )
        return {
            **contact_history,
            "walk_landing_x": landing_x,
            "walk_swing_seen": swing_seen,
            "walk_forward_steps": steps,
            "walk_body_progress": body_progress,
            "walk_support_progress": support_progress,
            "walk_progress_potential": progress_potential,
            "walk_progress_delta": progress_potential - previous_potential,
            "walk_new_steps": new_steps,
            "walk_goal_ready": ready,
            "walk_goal_hold_time": jnp.where(
                ready, info["walk_goal_hold_time"] + self.dt, jnp.float32(0.0)
            ),
        }

    def _task_reached(self, ps, info):
        reached = self._box_reached(ps, info)
        if self._walk_requires_locomotion:
            reached = reached & info["walk_goal_ready"] & (
                info["walk_goal_hold_time"] + 1e-6 >= self._bcfg.walk_success_hold_time
            )
        return reached

    def _walk_foot_target(self, info):
        """Foot-height objective: DIAL reference or the residual-mode CPG."""
        cfg = self._bcfg
        t = jnp.asarray(
            info.get("walk_reference_step", info["step"]), jnp.float32
        ) * self.dt
        if cfg.walk_gait_reference == "legacy":
            duty, cad, amp = self._gait_params[self._gait]
            return get_foot_step(duty, cad, amp, self._gait_phase[self._gait], t)
        phases = self._walk_phases(info)
        swing_progress = jnp.clip(phases / cfg.gait_swing_frac, 0.0, 1.0)
        lift = jnp.sin(jnp.pi * swing_progress) * (phases < cfg.gait_swing_frac)
        ramp = jnp.clip(t / max(float(cfg.gait_ramp_time), 1e-6), 0.0, 1.0)
        return self._feet_home[:, 2] + cfg.walk_foot_lift * ramp * lift

    def _walk_gait_error(self, ps, info):
        """Use actual foot clearance for DIAL-style absolute-joint search.

        A toe/heel pivot can raise the foot site while its capsule remains on
        the floor.  The reward must not count that rotation as swing clearance.
        """
        target = self._walk_foot_target(info)
        if self._walk_requires_locomotion and self._bcfg.walk_leg_control == "joint_target":
            if self._bcfg.walk_gait_reference == "cpg":
                # The CPG reference is expressed in the old foot-site frame;
                # physical floor clearance has no home-site height offset.
                target = target - self._feet_home[:, 2]
            measured = self._walk_contact_diagnostics(ps)["foot_floor_clearance"]
        else:
            measured = ps.site_xpos[self._feet_site_id][:, 2]
        return target - measured

    def _walk_nominal_stance(self, info):
        if self._bcfg.walk_leg_control == "joint_target":
            # Absolute joint search is not phase-forced by a CPG.  Use the
            # same nominal foot reference as its reward, including DIAL's
            # different left/right phase convention.  This is NOT an actual
            # support certificate; physical contact is checked separately.
            reference = self._walk_foot_target(info)
            ground = (0.0 if self._bcfg.walk_gait_reference == "legacy"
                      else self._feet_home[:, 2])
            return reference <= ground + 1e-6
        return self._walk_phases(info) >= self._bcfg.gait_swing_frac

    def _walk_phase_features(self, info):
        """Expose the reward clock to a joint-target policy without a hidden CPG."""
        cfg = self._bcfg
        cadence = (self._gait_params[self._gait][1]
                   if cfg.walk_gait_reference == "legacy" else cfg.gait_cadence)
        phase = (
            2.0 * jnp.pi * cadence * self.dt
            * jnp.asarray(
                info.get("walk_reference_step", info["step"]), jnp.float32
            )
        )
        return jnp.asarray([jnp.sin(phase), jnp.cos(phase)], jnp.float32)

    def _walk_reference_phase_anchor(self, ps, info):
        """Select the closest measured DIAL phase in the backward window.

        A fixed rewind can jump from the measured loaded gait into an unrelated
        support phase.  Compare the actual normalized planner joints with the
        already hash-locked DIAL reference and choose only among the current
        frame and its configured preceding window.  This cannot look ahead or
        change the benchmark clock; it only makes the task-owned UNLOAD entry
        continuous with the state that will execute it.
        """
        reference_step = jnp.asarray(info["walk_reference_step"], jnp.int32)
        rewind = int(self._bcfg.emergency_reference_rewind_steps)
        offsets = jnp.arange(rewind + 1, dtype=jnp.int32)
        indices = jnp.clip(
            reference_step - offsets,
            jnp.int32(0),
            jnp.int32(self._walk_joint_reference.shape[0] - 1),
        )
        joints = ps.qpos[
            jnp.asarray(self._robot_binding.qpos_indices[:self._n_planner])
        ]
        measured = self._whole_body_controller.planner_action_from_joints(
            joints
        )
        errors = jnp.mean(
            jnp.square(self._walk_joint_reference[indices] - measured), axis=-1
        )
        selected = jnp.argmin(errors)
        return indices[selected], jnp.sqrt(errors[selected])

    def _next_walk_reference_clock(self, info, phase_anchor=None):
        """Advance gait time or apply a measured monotone phase governor."""
        reference_step = jnp.asarray(info["walk_reference_step"], jnp.int32)
        recovery_anchor = jnp.asarray(
            info.get("walk_reference_recovery_anchor", 0), jnp.int32
        )
        mode = self._mga_inspection_mode(info)
        committed = jnp.asarray(info.get("mga_execution_mode", 0), jnp.int32)
        entering_unload = (mode == 1) & (committed == 0)
        rewind = jnp.int32(self._bcfg.emergency_reference_rewind_steps)
        proposed_anchor = (
            jnp.maximum(reference_step - rewind, jnp.int32(0))
            if phase_anchor is None
            else jnp.asarray(phase_anchor, jnp.int32)
        )
        # Repeated NORMAL/UNLOAD switches must not accumulate rewinds.  Keep
        # the largest previously certified recovery phase; later episodes may
        # move it forward but can never drag the gait reference backward.
        next_anchor = jnp.where(
            entering_unload,
            jnp.maximum(recovery_anchor, proposed_anchor),
            recovery_anchor,
        )
        next_step = jnp.where(
            mode == 1,
            jnp.where(entering_unload, next_anchor, reference_step),
            reference_step + jnp.int32(1),
        )
        return next_step, next_anchor

    def _next_walk_reference_step(self, info, phase_anchor=None):
        """Compatibility helper for tests and non-mutating clock inspection."""
        return HumanoidBoxPushEnv._next_walk_reference_clock(
            self, info, phase_anchor
        )[0]

    def step(self, state: State, action: jax.Array) -> State:
        if self._supports_mga_execution_context:
            state = self._mga_interval_state(state)
        cfg = self._bcfg
        record_physics = str(cfg.robot).lower() == "h1" and cfg.fast_force_loop
        physical_samples = None
        if cfg.fast_force_loop:
            # Same hierarchy as the arm force-control task: the MPC command is held for one
            # 20ms control interval, while impedance, stance PD and measured-force PI are
            # recomputed at every 4ms physics substep.
            def _substep(carry, _):
                ps_i, force_int = carry
                info_i = {**state.info, "force_int": force_int}
                tau_i = self._control(ps_i, action, info_i)
                ps_i = self._pipeline.step(self.sys, ps_i, tau_i, self._debug)
                force_int = self._update_force_integral(ps_i, action, info_i)
                sample = self._physics_safety_sample(ps_i) if record_physics else None
                return (ps_i, force_int), sample

            (ps, force_int), physical_samples = jax.lax.scan(
                _substep,
                (state.pipeline_state, state.info["force_int"]),
                (),
                self._n_frames,
            )
        else:
            tau = self._control(state.pipeline_state, action, state.info)
            ps = self.pipeline_step(state.pipeline_state, tau)
            force_int = self._update_force_integral(ps, action, state.info)
        return self._finish_step(state, ps, action, force_int, physical_samples)

    def _update_force_integral(self, ps, action, info):
        """The same measured-force PI update for every physics substep."""
        cfg = self._bcfg
        desired = self._hand_contact(ps, action, info)["F_n"]
        measured = self._box_contact_force(ps)
        proposed = jnp.clip(
            info["force_int"] + cfg.ki_force * (desired - measured),
            -cfg.force_int_max, cfg.force_int_max,
        )
        # Do not wind up while the moving box is outside the hands' workspace.
        updated = jnp.where(measured >= cfg.contact_acquire_force, proposed,
                            0.9 * info["force_int"])
        if self._supports_mga_execution_context:
            updated = jnp.where(self._mga_inspection_mode(info) == 1,
                                jnp.minimum(updated, 0.0), updated)
        return updated

    def _finish_step(self, state, ps, action, force_int, physical_samples=None):
        """Task state transition, separate from the physics implementation."""
        info = dict(state.info)
        records_physics = str(self._bcfg.robot).lower() == "h1"
        if records_physics:
            # Native/non-fast callers do not supply a physics tape.  They may
            # retain endpoint diagnostics, but must not invent substep coverage.
            info.update(self._empty_physics_safety_samples())
            if physical_samples is not None:
                info.update(physical_samples)
                info["physics_samples_valid"] = jnp.bool_(True)
        if self._walk_requires_locomotion:
            info.update(self._walk_progress(ps, state.info))
        reward, done = self._reward_done(ps, action, info)
        if (self._bcfg.walk_objective_mode == "dial"
                and self._bcfg.w_walk_effort > 0.0):
            # Match the applied first-substep torque, not a post-state control
            # or an unavailable interval average. Absorbing padding stays below.
            reward = jnp.nan_to_num(
                reward + self._walk_effort_penalty(
                    state.pipeline_state, action, state.info),
                nan=-1e3, posinf=-1e3, neginf=-1e3,
            )
        info["step"] = state.info["step"] + 1
        if "walk_reference_step" in state.info:
            phase_anchor, phase_error = self._walk_reference_phase_anchor(
                ps, state.info
            )
            next_reference, next_anchor = self._next_walk_reference_clock(
                state.info, phase_anchor,
            )
            info["walk_reference_step"] = next_reference
            info["walk_reference_recovery_anchor"] = next_anchor
            info["walk_reference_phase_error"] = phase_error
        info["force_int"] = force_int
        measured = self._box_contact_force(ps)
        detected = self._contact_acquisition_detected(ps, action, state.info, measured)
        just_acquired = ((state.info["contact_acquired"] < 0.5)
                         & detected)
        info["contact_acquired"] = jnp.maximum(
            state.info["contact_acquired"], detected.astype(jnp.float32))
        info["contact_step"] = jnp.where(just_acquired, info["step"], state.info["contact_step"])
        if str(self._bcfg.level).lower() == "unjam":
            released_now = (
                self._corridor_clearance(ps)
                >= self._bcfg.unjam_release_clearance
            )
            info["unjam_released"] = jnp.maximum(
                state.info["unjam_released"], released_now.astype(jnp.float32)
            )
        # A fall has priority over reaching the line, including later calls
        # after a failed transition in a fixed-length diagnostic rollout.
        fallen = jnp.maximum(state.info["task_fallen"], self._has_fallen(ps).astype(jnp.float32))
        if records_physics:
            # A transient fall inside an actually executed interval is terminal
            # even if its last substep recovers.  This is deliberately stricter
            # termination, using the existing fall thresholds unchanged.
            substep_fall = info["physics_samples_valid"] & jnp.any(
                info["physics_safety_margins"][:, 3] > 0.0
            )
            fallen = jnp.maximum(fallen, substep_fall.astype(jnp.float32))
        info["task_fallen"] = fallen
        reached = self._task_reached(ps, info) & (fallen < 0.5)
        info["task_success"] = jnp.maximum(
            state.info["task_success"], reached.astype(jnp.float32))
        info["prev_action"] = jnp.asarray(action, jnp.float32)
        # Production reset states always carry this realization memory.  Keep
        # compatibility with externally constructed/legacy states by never
        # changing the info pytree structure in the middle of a JAX scan.
        if "prev_realized_stiffness_raw" in state.info:
            info["prev_realized_stiffness_raw"] = jnp.asarray(
                action[self.spec.s_slice], jnp.float32
            )
        # Direct MPC diagnostics deliberately keep a fixed rollout length.  A completed task is
        # nevertheless terminal: make it absorbing so extra calls after ``done`` cannot create a
        # later fall or move the box beyond the line.
        terminal = state.info["task_success"] > 0.5
        ps = jax.tree_util.tree_map(
            lambda old, new: jnp.where(terminal, old, new),
            state.pipeline_state,
            ps,
        )
        # Brax training wrappers may attach nested metric dictionaries to
        # ``info``.  Preserve the complete pytree on an absorbing terminal;
        # scalar-only per-key ``where`` fails as soon as such a wrapper is
        # present even though ordinary evaluation states are flat.
        info = jax.tree_util.tree_map(
            lambda old, new: jnp.where(terminal, old, new),
            state.info,
            info,
        )
        info["step"] = state.info["step"] + 1
        # Only the tail AFTER completion is padding.  Its unexecuted actions
        # incur no objective penalty, while the true completion transition
        # retains its actual reward, contact force and physical risk.
        info["success_padding"] = terminal
        if records_physics:
            # The simulated suffix is discarded by the absorbing transition.
            # Neither its samples nor the last real interval may masquerade as
            # newly executed physics. Frozen endpoint risk remains available.
            empty = self._empty_physics_safety_samples()
            for key in empty:
                info[key] = jnp.where(terminal, empty[key], info[key])
        reward = jnp.where(terminal, self._success_padding_reward(), reward)
        done = jnp.where(terminal, jnp.float32(1.0), jnp.maximum(done, fallen))
        if self._supports_mga_execution_context:
            info = self._mga_finish_context(state.info, info, terminal)
        obs = self._get_obs(ps, info)
        return state.replace(pipeline_state=ps, obs=obs, reward=reward, done=done, info=info)

    def _has_fallen(self, ps):
        up = brax_math.rotate(
            jnp.array([0.0, 0.0, 1.0]), ps.x.rot[self._torso_idx - 1]
        )[2]
        return (up < 0.0) | (ps.x.pos[self._torso_idx - 1, 2] < 0.6 * self._torso_z0)

    def _safety_margins(self, ps, forces, balance_residual):
        """One signed contract for the existing force/contact/balance/fall checks.

        The two-foot balance proxy is retained conservatively; it is not a
        certificate of actual dynamic support.  Positive entries are unsafe.
        The nonhand threshold matches the sequence-risk contact event, rather
        than making every collision-free state sit exactly on that boundary.
        """
        cfg = self._bcfg
        up = brax_math.rotate(
            jnp.array([0.0, 0.0, 1.0]), ps.x.rot[self._torso_idx - 1]
        )[2]
        height_limit = 0.6 * self._torso_z0
        return jnp.asarray([
            (forces["hand"] - cfg.f_max) / max(float(cfg.f_max), 1.0e-6),
            (forces["nonhand"] - 0.5) / max(float(cfg.f_max), 1.0e-6),
            balance_residual / max(float(cfg.support_radius), 1.0e-6),
            jnp.maximum(-up, (height_limit - ps.x.pos[self._torso_idx - 1, 2])
                        / max(float(height_limit), 1.0e-6)),
        ])

    def _balance_safety_residual(self, ps):
        """Fixed-support CoM margin or P4's dynamic posture envelope.

        The original DIAL gait deliberately moves the pelvis away from the
        two-foot midpoint during supported exchange.  Treating that static
        proxy as a zero-tolerance safety condition makes a valid walk trigger
        UNLOAD.  Strict locomotion instead uses the model-predicted torso-up
        and height margins, while non-hand collision and terminal fall remain
        independent hard rows in :meth:`_safety_margins`.
        """
        if self._walk_requires_locomotion:
            up = brax_math.rotate(
                jnp.array([0.0, 0.0, 1.0]), ps.x.rot[self._torso_idx - 1]
            )[2]
            height = ps.x.pos[self._torso_idx - 1, 2]
            up_margin = (
                jnp.float32(self._bcfg.walk_safety_min_torso_up) - up
            ) * self._bcfg.support_radius
            height_margin = (
                jnp.float32(self._bcfg.walk_safety_min_height_ratio) * self._torso_z0
                - height
            ) / max(float(self._torso_z0), 1.0e-6) * self._bcfg.support_radius
            return jnp.maximum(up_margin, height_margin)
        feet_xy = ps.site_xpos[self._feet_site_id, :2].mean(axis=0)
        return (jnp.linalg.norm(
            ps.x.pos[self._pelvis_idx - 1, :2] - feet_xy
        ) - self._bcfg.support_radius)

    def _walk_terminal_recovery_residual(self, ps):
        """P4 terminal set strictly inside the physical posture envelope.

        A state can satisfy the hard torso-up/height boundary while already
        carrying enough angular momentum that no two-interval unload remains
        safe.  Normal receding candidates therefore end inside the midpoint
        between the nominal upright posture and each hard P4 boundary.  This
        task-owned inner set is used only at the model horizon terminal state;
        task-owned emergency candidates retain the unchanged physical set.
        """
        cfg = self._bcfg
        up = brax_math.rotate(
            jnp.array([0.0, 0.0, 1.0]), ps.x.rot[self._torso_idx - 1]
        )[2]
        height_ratio = (
            ps.x.pos[self._torso_idx - 1, 2]
            / max(float(self._torso_z0), 1.0e-6)
        )
        recovery_up = 0.5 * (1.0 + float(cfg.walk_safety_min_torso_up))
        recovery_height = 0.5 * (
            1.0 + float(cfg.walk_safety_min_height_ratio)
        )
        return jnp.maximum(
            jnp.float32(recovery_up) - up,
            jnp.float32(recovery_height) - height_ratio,
        )

    def _empty_physics_safety_samples(self):
        """Fixed-shape storage, explicitly invalid until physics is observed."""
        return {
            "physics_hand_force": jnp.zeros((self._n_frames,), jnp.float32),
            "physics_nonhand_force": jnp.zeros((self._n_frames,), jnp.float32),
            "physics_safety_margins": jnp.zeros((self._n_frames, 4), jnp.float32),
            "physics_samples_valid": jnp.bool_(False),
        }

    def _physics_safety_sample(self, ps):
        """One post-physics sample; no control, reward or contact-chart changes."""
        forces = self._box_contact_forces(ps)
        return {
            "physics_hand_force": forces["hand"],
            "physics_nonhand_force": forces["nonhand"],
            "physics_safety_margins": self._safety_margins(
                ps, forces, self._balance_safety_residual(ps)
            ),
        }

    def _transition_safety_margins(self, state, endpoint_margins=None):
        """Executed-interval envelope shared by candidate risk and metrics.

        Unknown active H1 coverage returns +inf (not an observed violation and
        not a reliability training label). Reset and unexecuted success padding
        use the frozen/current endpoint without duplicating an old interval.
        """
        if endpoint_margins is None:
            ps = state.pipeline_state
            forces = self._box_contact_forces(ps)
            _, g = self._manifold(ps, jnp.zeros((self.action_size,), jnp.float32), state.info)
            endpoint_margins = self._safety_margins(ps, forces, g[0])
        if str(self._bcfg.robot).lower() != "h1":
            return endpoint_margins
        valid = jnp.asarray(state.info.get("physics_samples_valid", False), jnp.bool_)
        padding = jnp.asarray(state.info.get("success_padding", False), jnp.bool_)
        reset = jnp.asarray(state.info.get("step", 0)) == 0
        samples = state.info.get("physics_safety_margins", jnp.zeros((self._n_frames, 4)))
        unknown = jnp.full((4,), jnp.inf, jnp.float32)
        envelope = jnp.maximum(endpoint_margins, jnp.max(samples, axis=0))
        envelope = jnp.where(jnp.all(jnp.isfinite(envelope)), envelope, unknown)
        return jnp.where(padding | reset, endpoint_margins, jnp.where(valid, envelope, unknown))

    def _success_padding_reward(self):
        """No-penalty continuing goal value; not an arbitrary terminal bonus."""
        cfg = self._bcfg
        return jnp.float32(cfg.w_alive + cfg.w_prog * cfg.push_dist)

    def _walk_velocity_target(self, info):
        """Body-frame command using the pre-transition physical clock."""
        cfg = self._bcfg
        elapsed = jnp.asarray(info["step"], jnp.float32) * self.dt
        target = jnp.float32(cfg.target_vx)
        return jnp.asarray([
            jnp.minimum(target * elapsed / float(cfg.walk_velocity_ramp_time), target),
            0.0,
        ], jnp.float32)

    def _walk_effort_penalty(self, pre_ps, action, info):
        """Normalized effort of the first actually applied control torque."""
        torque = self._control(pre_ps, action, info)
        return -jnp.float32(self._bcfg.w_walk_effort) * jnp.sum(
            (torque / self.joint_torque_range[:, 1]) ** 2
        )

    # --- reward (= -J_hum reduced) ---
    def _walk_coasting_box_x(self, ps):
        """Conditional stopping-location heuristic, not a safety certificate.

        If hands unload immediately and no other x force acts, the planar
        box loses kinetic energy to its existing slide frictionloss. Continuing
        hand forces invalidate that coast prediction; actual task success and
        all physical risk checks remain independent of this objective.
        """
        x = ps.x.pos[self._box_idx - 1, 0]
        vx = ps.xd.vel[self._box_idx - 1, 0]
        return x + vx * jnp.abs(vx) / (2.0 * self._walk_coast_deceleration)

    def _reward_done(self, ps, action, info):
        cfg = self._bcfg
        x = ps.x
        box_pos = x.pos[self._box_idx - 1]
        box_x = box_pos[0]
        box_yaw = brax_math.quat_to_euler(x.rot[self._box_idx - 1])[2]
        r_box = -((box_x - info["box_goal_x"]) ** 2)       # box reaches the goal LINE x
        if cfg.walk_box_goal_mode == "coast":
            r_box = -((self._walk_coasting_box_x(ps) - info["box_goal_x"]) ** 2)
        # Progress saturates at the line.  The previous unbounded linear term
        # paid the controller to keep pushing forever and analytically moved
        # the reward optimum far beyond the requested goal.
        r_prog = jnp.clip(box_x - info["box_x0"], 0.0, cfg.push_dist)
        r_align = -(box_pos[1] ** 2 + (self._align_yaw * box_yaw) ** 2)   # straight push (L3 frees yaw)
        vec = brax_math.rotate(jnp.array([0.0, 0.0, 1.0]), x.rot[self._torso_idx - 1])
        r_upright = -jnp.sum((vec - jnp.array([0.0, 0.0, 1.0])) ** 2)
        if cfg.walk_objective_mode == "dial":
            # Vendor H1 uses x.rot[0], semantically the pelvis (lines 491-494).
            pelvis_up = brax_math.rotate(
                jnp.array([0.0, 0.0, 1.0]), x.rot[self._pelvis_idx - 1])
            r_upright = -jnp.sum((pelvis_up - jnp.array([0.0, 0.0, 1.0])) ** 2)
        h, g = self._manifold(ps, action, info)
        contact = self._hand_contact(ps, action, info)
        box_forces = self._box_contact_forces(ps)
        force_scale = max(float(cfg.f_max), 1e-6)
        # The benchmark owns the desired force.  Scoring realized force against
        # the action-selected command lets a baseline choose zero force and earn
        # a false tracking advantage.  MGA and DIAL therefore share this same
        # time-indexed target ramp in both reward and evaluation.
        force_ref = self.requested_force_reference(ps, info)
        r_force = -((box_forces["hand"] - force_ref) / force_scale) ** 2
        r_force_limit = -(jnp.maximum(box_forces["hand"] - cfg.f_max, 0.0) / force_scale) ** 2
        nonhand_ratio_sq = (box_forces["nonhand"] / force_scale) ** 2
        if cfg.walk_objective_mode == "dial":
            # Vendor H1 push-crate scores unwanted contacts with a bounded
            # contact count.  Keep our force-aware signal, but prevent a
            # single catastrophic rollout from setting the reward scale for
            # the entire diffusion batch.  This changes only P4 ranking: the
            # physical 0.5 N violation margin and MGA gate remain unbounded.
            nonhand_ratio_sq = jnp.minimum(nonhand_ratio_sq, 1.0)
        r_nonhand = -nonhand_ratio_sq
        r_contact = -jnp.sum(h[2:5] ** 2)                  # hand-box contact residual (h_hand)
        r_bal = -jnp.maximum(g[0], 0.0) ** 2
        torso_z = x.pos[self._torso_idx - 1, 2]
        r_height = -(torso_z - self._torso_z0) ** 2        # stay near standing height (don't sink)
        if cfg.walk_objective_mode == "dial":
            r_height = -(torso_z - cfg.walk_height_target) ** 2
        fallen = self._has_fallen(ps)
        reached = self._task_reached(ps, info)
        done = (fallen | reached).astype(jnp.float32)
        r_alive = 1.0 - fallen.astype(jnp.float32)
        r_smooth = -jnp.sum(
            ps.qvel[jnp.asarray(self._robot_binding.dof_indices)] ** 2
        )   # anti-tremor: prefer calm joints
        r_stiffness_nominal = -jnp.mean(
            action[self.spec.s_slice] ** 2
        )
        r_yaw = -brax_math.quat_to_euler(x.rot[self._torso_idx - 1])[2] ** 2   # torso faces fwd (DIAL)
        r_angvel = -jnp.sum(ps.xd.ang[self._torso_idx - 1] ** 2)              # torso not spinning (DIAL)
        if cfg.walk_objective_mode == "dial":
            yaw = brax_math.quat_to_euler(x.rot[self._torso_idx - 1])[2]
            r_yaw = -jnp.atan2(jnp.sin(yaw), jnp.cos(yaw)) ** 2
            omega_body = global_to_body_velocity(
                ps.xd.ang[self._torso_idx - 1], x.rot[self._torso_idx - 1])
            # w_angvel is an explicit SI coefficient. The development value
            # (pi/180)^2 reproduces vendor lines 509-513 at zero target rate;
            # this is numerical equivalence, not a radians/degrees conversion.
            r_angvel = -omega_body[2] ** 2
        reward = (cfg.w_box * r_box + cfg.w_prog * r_prog + cfg.w_upright * r_upright
                  + cfg.w_height * r_height + cfg.w_alive * r_alive + cfg.w_contact * r_contact
                  + cfg.w_force * r_force + cfg.w_force_limit * r_force_limit
                  + cfg.w_nonhand * r_nonhand
                  + cfg.w_bal * r_bal + cfg.w_align * r_align + cfg.w_smooth * r_smooth
                  + cfg.w_stiffness_nominal * r_stiffness_nominal
                  + cfg.w_yaw * r_yaw + cfg.w_angvel * r_angvel)
        if str(cfg.level).lower() == "unjam":
            reward = reward - cfg.w_unjam_yaw * box_yaw ** 2
        # L1 lean-and-brace (w_brace/w_stagger 0 elsewhere): keep the CoM over the FEET (don't topple
        # onto the box) + reward a forward staggered stance so the rear leg braces the push.
        feet_x = ps.site_xpos[self._feet_site_id][:, 0]
        com_x = x.pos[self._pelvis_idx - 1, 0]
        r_brace = -jnp.maximum(com_x - jnp.max(feet_x), 0.0) ** 2
        r_stagger = -jnp.maximum(0.22 - (jnp.max(feet_x) - jnp.min(feet_x)), 0.0)
        reward = reward + cfg.w_brace * r_brace + cfg.w_stagger * r_stagger
        if self._is_walk:   # L2: + DIAL-style gait (feet track a stepping pattern) + forward walk
            r_gait = -jnp.sum(self._walk_gait_error(ps, info) ** 2)
            vb = global_to_body_velocity(ps.xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1])
            r_vel = -(vb[0] - cfg.target_vx) ** 2
            if cfg.walk_objective_mode == "dial":
                r_vel = -jnp.sum((vb[:2] - self._walk_velocity_target(info)) ** 2)
            reward = reward + cfg.w_gait * r_gait + cfg.w_vel * r_vel
            if self._walk_requires_locomotion:
                progress_scale = max(float(cfg.push_dist), 1e-6)
                r_walk_progress = jnp.asarray(
                    info.get("walk_progress_delta", 0.0), jnp.float32
                ) / progress_scale
                required_steps = max(2 * int(cfg.walk_min_steps_per_foot), 1)
                r_walk_step = jnp.asarray(
                    info.get("walk_new_steps", 0.0), jnp.float32
                ) / float(required_steps)
                reward = (
                    reward
                    + cfg.w_walk_progress * r_walk_progress
                    + cfg.w_walk_step * r_walk_step
                    - cfg.w_walk_fall * fallen.astype(jnp.float32)
                )
        # NaN-guard: a physics blow-up in an MGA rollout must score very low (rejected), not poison
        # the sample-weighted average (-> NaN action). H1 reward is always finite, so this is a no-op there.
        reward = jnp.nan_to_num(reward, nan=-1e3, posinf=-1e3, neginf=-1e3)
        return reward, done

    # --- full contact manifold (eq:humanoid_manifold) ---
    def _manifold(self, ps, action, info):
        cfg = self._bcfg
        x = ps.x
        c = self._hand_contact(ps, action, info)
        box_pos = x.pos[self._box_idx - 1]
        box_yaw = brax_math.quat_to_euler(x.rot[self._box_idx - 1])[2]
        # planar box: z + roll/pitch are fixed by the planar joint; the box equality is
        # STRAIGHT-PUSH ALIGNMENT — keep lateral y and yaw ~0 (L3 frees the yaw term).
        h_box = jnp.array([box_pos[1], self._align_yaw * box_yaw])     # box_y, box_yaw alignment
        h_hand = c["p_hand"] - c["p_c"]                    # hand on selected face (eq:hand_box_contact)
        h_hand_R = c["push_axis"] + c["n_c"]              # distal push axis to -face normal
        feet = ps.site_xpos[self._feet_site_id]
        # A walking task constrains only ground contact.  The fixed-stance force-control tasks also
        # anchor foot x/y to reset: without this distinction MGA can satisfy the old z-only
        # residual while swinging an ankle through the box.
        h_foot = (feet[:, 2] if self._is_walk else (feet - self._feet_home).reshape(-1))
        if self._walk_requires_locomotion:
            stance = self._walk_nominal_stance(info)
            # A swing foot must be allowed to leave the ground.  Keeping its
            # old ground equality active makes the optimizer oppose the CPG.
            if cfg.walk_leg_control == "joint_target":
                h_foot = self._walk_contact_diagnostics(ps)["foot_floor_clearance"] * stance
            else:
                h_foot = (feet[:, 2] - self._feet_home[:, 2]) * stance
        h = jnp.concatenate([h_box, h_hand, h_hand_R, h_foot])
        com_xy = x.pos[self._pelvis_idx - 1, :2]
        feet_c = feet.mean(axis=0)[:2]
        g_bal = self._balance_safety_residual(ps)                             # balance (eq:humanoid_ineq)
        g_fric = jnp.linalg.norm(c["f_t"]) - self._mu * c["f_n"]               # friction cone
        g_tip = jnp.abs(com_xy[0] - feet_c[0]) - cfg.support_radius            # forward tipping margin
        g = jnp.array([g_bal, g_fric, g_tip])
        return h, g

    def constraint_residual(self, state, action, ctx=None):
        return self._manifold(state.pipeline_state, action, state.info)

    def soft_feasibility_residual(self, state, action, ctx=None):
        """Task-owned AL residual, separate from physical constraint hooks.

        The physical constraint hook remains unchanged for ATACOM, diagnostics
        and risk certification, including unsafe contacts at completion.
        """
        h, g = self.constraint_residual(state, action, ctx)
        return self._soft_feasibility_from_constraints(state, h, g)

    def _soft_feasibility_from_constraints(self, state, h, g, forces=None):
        """Share sampling/acceptance scores without masking physical risks.

        H1 unjamming's virtual right-hand wrench is an approach command in
        free space, not a supported contact force.  Its friction-cone soft
        residual applies only during actual right-hand box compression.  The
        residual remains in newtons; neither its scale nor the physical force
        certificate changes.  Other tasks retain their historical AL rows.
        """
        if self._bcfg.robot.lower() == "h1" and self._bcfg.level.lower() == "unjam":
            if forces is None:
                forces = self._box_contact_forces(state.pipeline_state)
            g = g.at[1].set(jnp.where(forces["right_supported"], g[1], 0.0))
        padding = state.info["success_padding"]
        return jnp.where(padding, jnp.zeros_like(h), h), jnp.where(padding, jnp.zeros_like(g), g)

    # --- clean-state constraint-manifold geometry (primitive only; no mjx) ---
    # General env hooks (any manifold-aware solver may call them): the clean-state
    # constant-stiffness + force-target manifold; the physics-coupled contact / balance
    # go to the AL. ``manifold_residual`` is the flattened residual C(U) over a
    # node-control trajectory; ``manifold_geometry`` its tangent geometry ∂(½‖C‖²)/∂U.
    def _corridor_clearance(self, ps):
        """Signed lateral clearance of the yawed box inside the unjam corridor."""
        box_pos = ps.x.pos[self._box_idx - 1]
        box_yaw = brax_math.quat_to_euler(ps.x.rot[self._box_idx - 1])[2]
        projected_half_width = self._half * (
            jnp.abs(jnp.cos(box_yaw)) + jnp.abs(jnp.sin(box_yaw))
        )
        return (
            self._bcfg.unjam_corridor_half_width
            - jnp.abs(box_pos[1])
            - projected_half_width
        )

    def _manifold_res_node(self, u):
        s0, s1 = self.spec.s_slice.start, self.spec.s_slice.stop
        f_span = max(self._bcfg.f_max - self._bcfg.f_min, 1e-6)
        dS = self._bcfg.s_scale * u[s0:s1]
        F = self._force_cmd(u[self.spec.nu_slice][0])
        # Fixed-stance P1--P3 use the validated nominal stiffness chart.  P3
        # retains both bounded contact coordinates as genuine model-based
        # degrees of freedom; P4 instead leaves stiffness free for
        # dynamic walk/contact adaptation.
        residuals = []
        if not self._is_walk:
            residuals.append(dS)
        residuals.append(jnp.array([(F - self._bcfg.f_target) / f_span]))
        if self._face_select:
            j0 = 3 if self._bcfg.use_base else 0
            # The raw chart is centered at zero; _unpack maps that origin to
            # the reachable rear/off-centre physical contact while jammed.
            # Geometric release does not imply final yaw alignment.  Keep a
            # and b available to rollout/AL optimization on either side of
            # that event; neither coordinate is a clean safety equality.
            residuals.append(u[j0:j0 + 3])
        return jnp.concatenate(residuals)

    def manifold_residual(self, state, Ybar_nodes):
        del state
        return jax.vmap(self._manifold_res_node)(Ybar_nodes).reshape(-1)

    def manifold_geometry(self, state, Ybar_nodes, t0):
        del state, t0
        sq = lambda u: 0.5 * jnp.sum(self._manifold_res_node(u) ** 2)
        return jax.vmap(jax.grad(sq))(Ybar_nodes)

    def manifold_residual_horizon(self, state, dense_actions, t0):
        """Task-owned clean contact chart over the complete dense horizon."""
        del state, t0
        return jax.vmap(self._manifold_res_node)(dense_actions).reshape(-1)

    def prepare_realization_context(
        self, state, dense_actions, *, gate_controllability=False,
    ):
        """Freeze the measured reliability of the current contact mode.

        H1's semantic primitive is a direct chart (not an incremental pose
        command), so its clean horizon manifold does not need the arm tasks'
        coordinate-bias lead.  The measured gate still records whether each
        action block is locally executable and is consumed by the backend's
        component gate.
        """
        del dense_actions
        gate = (
            self.geometry_reliability(state)["action"]
            if gate_controllability else
            jnp.ones((self.action_size,), jnp.float32)
        )
        return state.replace(info={
            **state.info,
            "_mga_realization_gate": jax.lax.stop_gradient(gate),
        })

    def manifold_residual_horizon_controllable(self, state, dense_actions, t0):
        # The direct semantic chart is already expressed in its controllable
        # coordinates.  Measured controllability gates the update blocks in
        # ``geometry_reliability`` without weakening the clean equalities.
        return self.manifold_residual_horizon(state, dense_actions, t0)

    def geometry_reliability(self, state):
        ps = state.pipeline_state
        cfg = self._bcfg
        h, g = self._manifold(
            ps, jnp.zeros((self.action_size,), jnp.float32), state.info
        )
        forces = self._box_contact_forces(ps)
        contact = jnp.clip(forces["hand"] / max(cfg.f_target, 1.0), 0.0, 1.0)
        force_risk = jax.nn.relu(forces["hand"] / max(cfg.f_max, 1.0) - 0.8)
        balance_risk = jax.nn.relu(g[0]) / max(cfg.support_radius, 1.0e-6)
        invalid_risk = jnp.clip(forces["nonhand"] / max(cfg.f_max, 1.0), 0.0, 1.0)
        contact_error = jnp.linalg.norm(h[2:5]) / max(cfg.contact_acquire_gap, 1.0e-3)
        scalar = jnp.exp(-(
            force_risk + balance_risk + invalid_risk + 0.25 * contact_error
        ))
        # Keep the task-progress/contact coordinates available for recovery;
        # attenuate stiffness/feed and walking corrections under low trust.
        action_gate = jnp.ones((self.action_size,), jnp.float32)
        action_gate = action_gate.at[self.spec.s_slice].set(scalar)
        action_gate = action_gate.at[self.spec.nu_slice].set(scalar)
        if self._is_walk:
            action_gate = action_gate.at[self.spec.total_width:].set(scalar)
        result = {
            "scalar": scalar,
            "action": action_gate,
            "clean_action": jnp.ones_like(action_gate),
            # Shared MGA diagnostics schema.  H1 keeps task-progress/contact
            # coordinates open (path=1), has no independent sampled surface
            # normal (report measured contact confidence), and applies the
            # scalar gate to both stiffness and force action blocks.
            "path": jnp.float32(1.0),
            "normal": contact,
            "stiffness": scalar,
            "contact": contact,
            "force": scalar,
            "force_only": jnp.exp(-force_risk),
            "balance": jnp.exp(-balance_risk),
            "invalid_contact": jnp.exp(-invalid_risk),
            "contact_error": contact_error,
        }
        if self._walk_requires_locomotion and self._walk_joint_reference is not None:
            # Before measured contact, stiffness and joint residual effects
            # are not identifiable.  Preserve the verified DIAL reference and
            # nominal hand impedance instead of substituting Gaussian drift;
            # task/contact coordinates remain searchable.  Once contact is
            # measured, progressively expose the residual correction blocks.
            update_trust = contact * scalar
            update_gate = jnp.ones((self.action_size,), jnp.float32)
            update_gate = update_gate.at[self.spec.s_slice].set(update_trust)
            update_gate = update_gate.at[self.spec.nu_slice].set(update_trust)
            update_gate = update_gate.at[self.spec.total_width:].set(update_trust)
            result["update_action"] = update_gate
        return result

    def reliability_features(self, state, action):
        return self.reliability_features_sequence(state, action[None, :])

    def reliability_features_sequence(self, state, actions):
        """Observable fixed-width state/candidate features for P1--P4."""
        ps = state.pipeline_state
        cfg = self._bcfg
        forces = self._box_contact_forces(ps)
        box = ps.x.pos[self._box_idx - 1]
        yaw = brax_math.quat_to_euler(ps.x.rot[self._box_idx - 1])[2]
        feet = ps.site_xpos[self._feet_site_id]
        com = ps.x.pos[self._pelvis_idx - 1, :2]
        support = feet[:, :2].mean(axis=0)
        balance_feature = (
            self._balance_safety_residual(ps) / max(cfg.support_radius, 1.0e-6)
            if getattr(self, "_walk_requires_locomotion", False) else
            jnp.linalg.norm(com - support) / max(cfg.support_radius, 1.0e-6)
        )
        state_features = jnp.asarray([
            (box[0] - state.info["box_x0"]) / max(cfg.push_dist, 1.0e-6),
            (state.info["box_goal_x"] - box[0]) / max(cfg.push_dist, 1.0e-6),
            box[1] / max(float(self._half), 1.0e-6),
            yaw / 0.1,
            forces["hand"] / max(cfg.f_max, 1.0),
            forces["wall"] / max(cfg.f_max, 1.0),
            forces["nonhand"] / max(cfg.f_max, 1.0),
            balance_feature,
            state.info["contact_acquired"],
            state.info["unjam_released"],
            state.info["force_int"] / max(cfg.force_int_max, 1.0),
            self._corridor_clearance(ps) / max(cfg.unjam_release_clearance, 1.0e-3),
        ], jnp.float32)
        semantic = actions[:, :self.spec.total_width]
        previous = state.info["prev_action"][:self.spec.total_width]
        delta0 = semantic[0] - previous
        sequence_features = jnp.asarray([
            jnp.sqrt(jnp.mean(semantic[:, self.spec.r_slice] ** 2)),
            jnp.sqrt(jnp.mean(semantic[:, self.spec.s_slice] ** 2)),
            jnp.mean(semantic[:, self.spec.nu_slice]),
            jnp.max(jnp.abs(semantic[:, self.spec.nu_slice])),
            jnp.sqrt(jnp.mean(delta0 ** 2)),
            jnp.sqrt(jnp.mean(jnp.diff(semantic, axis=0) ** 2))
            if actions.shape[0] > 1 else jnp.float32(0.0),
            jnp.mean(semantic[:, 0]),
            jnp.mean(semantic[:, 3]),
            jnp.mean(semantic[:, 4]),
            jnp.sqrt(jnp.mean(actions[:, self.spec.total_width:] ** 2))
            if self._is_walk else jnp.float32(0.0),
            jnp.max(jnp.abs(actions[:, self.spec.total_width:]))
            if self._is_walk else jnp.float32(0.0),
            jnp.asarray(self._is_walk, jnp.float32),
        ], jnp.float32)
        return jnp.concatenate([state_features, sequence_features])

    def sequence_score_risk(self, state, actions, aug_lambda=0.0, aug_rho=0.0):
        if self._supports_mga_execution_context:
            state = self._prepare_mga_execution_state(state, 0)
        return self._sequence_score_risk(state, actions, aug_lambda, aug_rho)

    def _realized_transition_score_risk(
        self, next_state, action, aug_lambda, aug_rho,
    ):
        """Score one already-realized model transition without advancing MJX."""
        cfg = self._bcfg
        h, g = self.constraint_residual(next_state, action)
        forces = self._box_contact_forces(next_state.pipeline_state)
        h_soft, g_soft = self._soft_feasibility_from_constraints(
            next_state, h, g, forces
        )
        residual = jnp.concatenate([jnp.abs(h_soft), jax.nn.relu(g_soft)])
        penalty = (
            aug_lambda * jnp.sum(residual)
            + 0.5 * aug_rho * jnp.sum(residual ** 2)
        )
        endpoint_margins = self._safety_margins(
            next_state.pipeline_state, forces, g[0]
        )
        margins = self._transition_safety_margins(
            next_state, endpoint_margins
        )
        force_violation = (margins[0] > 0.0).astype(jnp.float32)
        invalid = (
            (margins[1] > 0.0) | (margins[3] > 0.0)
        ).astype(jnp.float32)
        balance = jax.nn.relu(margins[2])
        reference = self.requested_force_reference(
            next_state.pipeline_state, next_state.info
        )
        force_mae = (
            jnp.abs(forces["hand"] - reference) / max(cfg.f_target, 1.0)
        )
        return next_state.reward - penalty, jnp.asarray([
            force_violation, invalid, balance, force_mae,
        ])

    def _apply_initial_physics_risk(self, state, risk):
        if str(self._bcfg.robot).lower() != "h1":
            return risk
        initial = self._physics_safety_sample(
            state.pipeline_state
        )["physics_safety_margins"]
        initial_risk = jnp.asarray([
            (initial[0] > 0.0).astype(jnp.float32),
            ((initial[1] > 0.0) | (initial[3] > 0.0)).astype(jnp.float32),
            jax.nn.relu(initial[2]),
        ])
        initial_risk = jnp.where(
            jnp.all(jnp.isfinite(initial)), initial_risk,
            jnp.full((3,), jnp.inf),
        )
        return risk.at[:3].set(jnp.maximum(risk[:3], initial_risk))

    def _score_emergency_transition(
        self, prepared, predicted, action, aug_lambda, aug_rho,
    ):
        score, risk = self._realized_transition_score_risk(
            predicted, action, aug_lambda, aug_rho
        )
        return score, self._apply_initial_physics_risk(prepared, risk)

    def _aggregate_sequence_risks(self, risks):
        """Separate P4's recursive backup certificate from horizon prediction."""
        if self._walk_requires_locomotion:
            # The first row certifies the 20 ms interval deployed now; the
            # remaining configured rows provide a short viability buffer for
            # the shifted incumbent and task-owned emergency bank.  This avoids
            # treating the complete open-loop horizon as deployed while still
            # rejecting an action whose immediate successors already leave the
            # physical safe set. Hsample is positive by contract; min keeps the
            # helper well-defined for short diagnostic horizons.
            backup = min(
                int(getattr(
                    getattr(self, "_bcfg", None), "emergency_backup_steps", 2
                )),
                int(risks.shape[0]),
            )
            hard_risk = jnp.max(risks[:backup, :3], axis=0)
        else:
            tail = max(1, (int(risks.shape[0]) + 4) // 5)
            hard_risk = jnp.asarray([
                jnp.max(risks[:, 0]),
                jnp.max(risks[:, 1]),
                jnp.mean(jnp.sort(risks[:, 2])[-tail:]),
            ])
        return jnp.concatenate([hard_risk, jnp.mean(risks[:, 3])[None]])

    def _sequence_score_risk(self, state, actions, aug_lambda, aug_rho, execution_mode=0):
        def body(s, u):
            if self._supports_mga_execution_context:
                s = self._prepare_mga_execution_state(s, execution_mode)
            s2 = self.step(s, u)
            score, risk = self._realized_transition_score_risk(
                s2, u, aug_lambda, aug_rho
            )
            return s2, (score, risk)

        terminal_state, (scores, risks) = jax.lax.scan(body, state, actions)
        # Receding P4 deploys exactly one 20 ms control interval before
        # measuring and replanning.  Certifying an unchanged eight-step
        # suffix as if it would be executed open-loop can reject a safe first
        # action and destroy the support exchange.  Certifying only that first
        # interval can instead enter a state with no safe successor.  The P4
        # hard risk therefore covers the deployed interval and its first
        # shifted-incumbent backup, including every 4 ms MJX substep.  The
        # complete rollout still supplies score/force prediction, and fixed
        # tasks retain their full-horizon certificate.
        risk = self._aggregate_sequence_risks(risks)
        if self._walk_requires_locomotion and execution_mode == 0:
            # Physical safety of the first receding intervals is necessary
            # but not sufficient: the old certificate admitted a horizon
            # whose terminal state was safe yet already unrecoverable.  Keep
            # normal candidates inside a task-owned terminal recovery core.
            # UNLOAD/emergency rollouts use execution_mode=1 and deliberately
            # retain only the unchanged physical boundary.
            terminal_recovery = jax.nn.relu(
                self._walk_terminal_recovery_residual(
                    terminal_state.pipeline_state
                )
            )
            risk = risk.at[2].set(jnp.maximum(risk[2], terminal_recovery))
        # The initial instantaneous state is not the previous interval's
        # envelope. A recovery from an unsafe state is never recertified as
        # having been safe throughout the candidate window.
        risk = self._apply_initial_physics_risk(state, risk)
        return jnp.mean(scores), risk

    def sequence_risk(self, state, actions):
        return self.sequence_score_risk(state, actions)[1]

    def sequence_risk_is_safe(self, risk):
        tolerance = 0.0 if str(self._bcfg.robot).lower() == "h1" else 1.0e-8
        return (risk[0] <= tolerance) & (risk[1] <= tolerance) & (risk[2] <= tolerance)

    def sequence_risk_is_no_worse(self, candidate, incumbent, tolerance):
        return jnp.all(candidate[:3] <= incumbent[:3] + tolerance[:3])

    def emergency_sequence_score_risk(self, state, actions, aug_lambda=0.0, aug_rho=0.0):
        if not self._supports_mga_execution_context:
            return self.sequence_score_risk(state, actions[:1], aug_lambda, aug_rho)
        prepared = self._prepare_mga_execution_state(state, 1)
        return self._sequence_score_risk(
            prepared, actions[:2], aug_lambda, aug_rho, execution_mode=1
        )

    def project_mga_candidate(self, state, nodes):
        projected = jnp.clip(nodes, -1.0, 1.0)
        if not (
            self._walk_requires_locomotion
            and self._bcfg.walk_leg_control == "joint_target"
            and self._walk_joint_reference is not None
        ):
            return projected

        # The shared time-indexed reference is the verified locomotion prior.
        # P4's contact model predicts the hand/box interaction, but it does not
        # support unconstrained whole-body gait search.  Preserve only the same
        # bounded local joint residual available to the learned policy prior;
        # Gaussian/refined candidates therefore cannot obtain extra authority.
        # Residual authority is additionally restricted to a measured support
        # exchange.  Both legs remain available while either leg is in a
        # verified swing: the policy can maintain the opposite support as it
        # corrects landing.  Double support and the torso stay on the shared
        # reference, so a short-horizon progress score cannot move the body
        # ahead of support between exchanges.
        # The full model certificate and terminal recovery core still decide
        # whether any such swing correction is executable.
        #
        # Task-owned emergency plans are different, but their identity is an
        # explicit backend candidate source and execution mode.  Do not infer
        # that mode here from a numeric force coordinate: a perfectly ordinary
        # terminal-coast NORMAL plan can share the same value.  The backend
        # bypasses this NORMAL proposal tube only for the explicit emergency
        # bank or a shifted incumbent whose committed mode is UNLOAD, and
        # still applies the solver-wide action bound.
        residual_limit = jnp.float32(
            self._bcfg.policy_joint_reference_residual_scale
        )
        swing_seen = jnp.asarray(
            state.info.get("walk_swing_seen", jnp.zeros((2,), jnp.bool_)),
            jnp.float32,
        )
        paired_leg_width = (self._n_planner - 1) // 2
        exchange_active = jnp.any(swing_seen).astype(jnp.float32)
        swing_mask = jnp.concatenate([
            jnp.full((paired_leg_width,), exchange_active, jnp.float32),
            jnp.full((paired_leg_width,), exchange_active, jnp.float32),
            jnp.zeros(
                (self._n_planner - 2 * paired_leg_width,), jnp.float32
            ),
        ])
        normal = projected.at[..., self.spec.total_width:].set(
            jnp.clip(
                projected[..., self.spec.total_width:],
                -residual_limit,
                residual_limit,
            ) * swing_mask
        )
        return normal

    def mga_prior_applicable(self, state):
        """Declare the task state in which a learned P4 correction is valid.

        The external DIAL reference and model-based refinement own gait
        acquisition.  Before one physically supported landing per foot, a
        short-horizon learned correction can improve box/contact score while
        preventing the opposite touchdown just beyond the horizon.  That is
        an out-of-support use of the loaded-contact prior, not evidence that
        the candidate is unsafe at the current 20 ms interval.  Enable the
        prior only after the task memory records a bilateral support exchange;
        all non-P4 tasks retain their existing always-applicable behavior.
        """
        if not self._walk_requires_locomotion:
            return jnp.asarray(True)
        steps = jnp.asarray(
            state.info.get("walk_forward_steps", jnp.zeros((2,), jnp.int32))
        )
        required = max(int(self._bcfg.walk_min_steps_per_foot), 1)
        return jnp.all(steps >= required)

    def _walk_roll_capture_residual(self, state):
        """Map DIAL's measured lateral capture correction into the leg chart."""
        residual = jnp.zeros((self._n_planner,), jnp.float32)
        ps = state.pipeline_state
        support_y = self._support_feedback(
            ps, self._foot_contact_loads(ps)
        )["support_reference_xy"][1]
        pelvis = self._pelvis_idx - 1
        capture_rad = jnp.clip(
            self._bcfg.gait_roll_capture_gain * (
                ps.x.pos[pelvis, 1]
                - support_y
                + self._bcfg.gait_roll_capture_lead * ps.xd.vel[pelvis, 1]
            ),
            -self._bcfg.gait_roll_limit,
            self._bcfg.gait_roll_limit,
        )
        planner = tuple(self._robot_profile.joint_groups["planner"])
        bounds = jnp.asarray(
            self._whole_body_controller.planner_joint_bounds, jnp.float32
        )
        reference_scale = jnp.float32(
            self._bcfg.walk_joint_reference_residual_scale
        )
        for hip in self._roll_hips:
            hip_in_planner = planner.index(hip)
            radians_per_action = (
                0.5
                * (bounds[hip_in_planner, 1] - bounds[hip_in_planner, 0])
                * reference_scale
            )
            residual = residual.at[hip_in_planner].set(jnp.clip(
                capture_rad
                / jnp.maximum(radians_per_action, jnp.finfo(jnp.float32).eps),
                -1.0,
                1.0,
            ))
        return residual

    def _walk_pitch_momentum_recovery_residual(self, state):
        """Map measured torso pitch momentum into the DIAL hip chart.

        UNLOAD can leave the torso moving through the nominal DIAL posture even
        when its current pose is physically safe.  Extrapolate that measured
        pitch rate over the existing gait capture lead time, then express the
        resulting bounded joint correction in the same normalized residual
        coordinates used by the external reference.  This is a recovery
        proposal only: the model-based NORMAL certificate remains responsible
        for accepting or rejecting it.
        """
        residual = jnp.zeros((self._n_planner,), jnp.float32)
        pitch_rate = state.pipeline_state.xd.ang[self._torso_idx - 1, 1]
        capture_rad = (
            jnp.float32(self._bcfg.gait_capture_lead) * pitch_rate
        )
        planner = tuple(self._robot_profile.joint_groups["planner"])
        bounds = jnp.asarray(
            self._whole_body_controller.planner_joint_bounds, jnp.float32
        )
        reference_scale = jnp.float32(
            self._bcfg.walk_joint_reference_residual_scale
        )
        for hip, _, _ in self._sag_legs:
            hip_in_planner = planner.index(hip)
            radians_per_action = (
                0.5
                * (bounds[hip_in_planner, 1] - bounds[hip_in_planner, 0])
                * reference_scale
            )
            residual = residual.at[hip_in_planner].set(jnp.clip(
                capture_rad
                / jnp.maximum(radians_per_action, jnp.finfo(jnp.float32).eps),
                -1.0,
                1.0,
            ))
        return residual

    def emergency_plan(self, state, reference_nodes, t0=0.0):
        """Task-owned UNLOAD preserves measured entry stiffness; legacy uses its old hold."""
        del t0
        emergency = jnp.zeros_like(reference_nodes)
        emergency = emergency.at[:, self.spec.nu_slice].set(-1.0)
        # H1 seals the actually applied matrix/raw coordinates at entry. A
        # zero feedforward coordinate alone does not identify execution mode.
        # Legacy robots retain their diagonal-only reduction; lowering every
        # svec entry introduces cross-axis coupling and can increase an eigenvalue.
        if self._supports_mga_execution_context:
            prepared = self._prepare_mga_execution_state(state, 1)
            emergency = emergency.at[:, self.spec.s_slice].set(
                prepared.info["mga_unload_stiffness_raw"])
        else:
            diag = self.spec.s_slice.start + jnp.asarray([0, 3, 5])
            emergency = emergency.at[:, diag].set(-1.0)
        if self._is_walk and self._bcfg.walk_leg_control == "joint_target":
            # With a shared time-indexed DIAL reference, zero residual keeps
            # the nominal gait continuous.  Reprojecting the measured pose
            # into its deliberately small residual tube saturates many joints
            # and creates a discontinuous full-body brake at contact.  Without
            # a reference, retain the legacy nearest representable hold.
            emergency = self._initialize_joint_target_plan(
                state, emergency,
                hold_current=getattr(self, "_walk_joint_reference", None) is None,
            )
            if getattr(self, "_walk_joint_reference", None) is not None:
                # Apply the measured lateral capture law from the project CPG;
                # the backend still rejects this UNLOAD candidate unless its
                # physical two-interval and NORMAL-successor checks both pass.
                emergency = emergency.at[:, self.spec.total_width:].set(
                    self._walk_roll_capture_residual(state)
                )
                # The large DIAL-style crate can enter the nominal foot-swing
                # envelope after the hands unload.  Flex both knees inside the
                # task's bounded residual chart so the recursive successor
                # keeps real foot geometry clear of the box.  Selecting the
                # currently leading foot is insufficient: the opposite foot
                # can be the one advanced by the next reference sample.
                planner = tuple(self._robot_profile.joint_groups["planner"])
                for _, knee, _ in self._sag_legs:
                    knee_in_planner = planner.index(knee)
                    emergency = emergency.at[
                        :, self.spec.total_width + knee_in_planner
                    ].set(1.0)
        return emergency

    def emergency_plans(self, state, reference_nodes, t0=0.0):
        """Return certified alternatives for P4's gait-phase-dependent unload.

        Other task modes retain the original one-plan contract.  Strict H1 P4
        combines only the two sagittal knee residuals; the hand unload,
        stiffness seal, DIAL reference, horizon and all safety limits remain
        identical.  The MGA backend evaluates every row from the measured
        state and may execute one only when its two-interval physical risk is
        safe.
        """
        nominal = self.emergency_plan(state, reference_nodes, t0)
        if not (
            self._is_walk
            and self._bcfg.walk_leg_control == "joint_target"
            and getattr(self, "_walk_joint_reference", None) is not None
        ):
            return nominal[None]
        planner = tuple(self._robot_profile.joint_groups["planner"])
        knee_coordinates = tuple(
            self.spec.total_width + planner.index(knee)
            for _, knee, _ in self._sag_legs
        )
        levels = tuple(float(value) for value in getattr(
            self._bcfg,
            "emergency_knee_residual_levels",
            (1.0, 0.0, -0.5, -1.0),
        ))
        candidates = []
        for left in levels:
            for right in levels:
                candidate = nominal.at[:, knee_coordinates[0]].set(left)
                candidate = candidate.at[:, knee_coordinates[1]].set(right)
                candidates.append(candidate)
        return jnp.stack(candidates, axis=0)

    def normal_recovery_plan(self, state, reference_nodes, t0=0.0):
        """Reference-following NORMAL proposal with continuous leg recovery.

        Returning blindly to a zero residual can leave a late support exchange
        just outside the terminal upright core even though the preceding
        UNLOAD is physically safe.  Reuse the existing DIAL gait's roll
        capture law to place both hips under the measured torso motion.  Keep
        the emergency knee-clearance residual continuous at the mode boundary,
        then taper it to the nominal reference across the recovery horizon.
        The backend still accepts this proposal only after the complete NORMAL
        horizon passes the unchanged model-based certificate.
        """
        del t0
        if not self._is_walk:
            # Fixed-stance tasks also need an explicit hybrid-mode exit. Once
            # an UNLOAD plan becomes the shifted incumbent, Gaussian
            # refinement alone is not a recovery contract: its centre remains
            # the zero-force sentinel and can keep the task unloaded forever.
            # Re-enter NORMAL continuously from the sealed entry stiffness and
            # ramp only the requested normal force over the node horizon. The
            # backend still checks the complete physical rollout before this
            # proposal can replace the UNLOAD incumbent.
            recovery = jnp.zeros_like(reference_nodes)
            stiffness = jnp.asarray(
                state.info["mga_unload_stiffness_raw"], recovery.dtype
            )
            recovery = recovery.at[:, self.spec.s_slice].set(stiffness)
            cfg = self._bcfg
            force_span = max(float(cfg.f_max - cfg.f_min), 1.0e-6)
            target_nu = jnp.clip(
                2.0 * (jnp.float32(cfg.f_target) - cfg.f_min) / force_span - 1.0,
                -1.0,
                1.0,
            )
            force_ramp = jnp.linspace(
                -1.0,
                target_nu,
                int(reference_nodes.shape[0]),
                dtype=recovery.dtype,
            )
            return recovery.at[:, self.spec.nu_slice].set(force_ramp[:, None])
        if not (
            self._is_walk
            and self._bcfg.walk_leg_control == "joint_target"
            and getattr(self, "_walk_joint_reference", None) is not None
            and self._bcfg.emergency_reference_rewind_steps > 0
        ):
            return None
        # Zero primitive coordinates restore the task's nominal force and
        # stiffness; zero joint residual follows the re-anchored external gait
        # reference.  The lateral correction deliberately uses the same
        # measured pelvis-over-support capture point as ``BipedalGait``.  Torso
        # roll is not that signal and can have the opposite sign during a
        # supported lateral translation.
        #
        # ``gait_roll_capture_gain`` is expressed in rad / m, whereas this
        # method returns the normalized residual around the external DIAL
        # reference.  Convert the physical correction through the exact
        # planner joint range and reference-residual scale before applying the
        # solver-wide [-1, 1] action bound.  The backend must still certify the
        # complete NORMAL horizon before this task-owned proposal may execute.
        recovery = jnp.zeros_like(reference_nodes)
        recovery = recovery.at[:, self.spec.total_width:].set(
            self._walk_roll_capture_residual(state)
        )
        planner = tuple(self._robot_profile.joint_groups["planner"])
        knee_coordinates = jnp.asarray([
            self.spec.total_width + planner.index(knee)
            for _, knee, _ in self._sag_legs
        ])
        # ``reference_nodes`` is the exact committed/considered UNLOAD plan,
        # so its first node is the only correct entry-side knee posture.  A
        # linear node-space fade gives the spline a continuous first control
        # while returning to zero residual at the certified horizon terminal.
        knee_entry = jnp.clip(
            reference_nodes[0, knee_coordinates], -1.0, 1.0
        )
        fade = jnp.linspace(
            1.0, 0.0, int(reference_nodes.shape[0]), dtype=recovery.dtype
        )
        return recovery.at[:, knee_coordinates].set(
            fade[:, None] * knee_entry[None, :]
        )

    def normal_recovery_plans(self, state, reference_nodes, t0=0.0):
        """Ordered P4 recovery bank with bounded momentum/roll release.

        The first candidate preserves the existing roll capture and continuous
        knee fade.  The second adds one early, spline-smoothed hip-pitch pulse
        derived from measured torso angular velocity.  Its first and terminal
        pitch nodes are unchanged, so it neither jumps at the UNLOAD/NORMAL
        boundary nor leaves a permanent pitch offset in the shared DIAL
        reference.

        A large measured lateral capture error can saturate both hip-roll
        residuals. Holding that saturated correction for the entire horizon
        can keep driving the torso after it has crossed the support point. The
        remaining ordered pairs preserve the exact entry node, then sweep the
        later hip-roll nodes through zero, half reversal and full reversal;
        each level is followed by its pitch-momentum combination.  This is a
        bounded anti-windup chart, symmetric in the measured capture sign, not
        a change to joint authority. MGA checks the bank sequentially and can
        execute only a fully revalidated member.
        """
        nominal = self.normal_recovery_plan(state, reference_nodes, t0)
        if nominal is None:
            return None
        if not self._is_walk:
            # Try the nominal target first, then two strictly less forceful
            # exits.  All candidates share the same sealed stiffness and
            # NORMAL contact realization; only the end of the monotone force
            # ramp changes.  A conservative exit is useful when the current
            # stance can safely reacquire geometry but cannot yet certify the
            # full requested load over one H16 window.
            full = nominal
            zero = nominal.at[:, self.spec.nu_slice].set(-1.0)
            half = nominal.at[:, self.spec.nu_slice].set(
                0.5 * (full[:, self.spec.nu_slice] - 1.0)
            )
            return jnp.stack([full, half, zero], axis=0)
        if int(nominal.shape[0]) < 3:
            return nominal[None]
        capture = nominal
        pitch_residual = self._walk_pitch_momentum_recovery_residual(state)
        capture = capture.at[1, self.spec.total_width:].add(pitch_residual)
        capture = jnp.clip(capture, -1.0, 1.0)
        planner = tuple(self._robot_profile.joint_groups["planner"])
        roll_coordinates = jnp.asarray([
            self.spec.total_width + planner.index(hip)
            for hip in self._roll_hips
        ])
        entry_sign = jnp.sign(nominal[0, roll_coordinates])
        candidates = [nominal, capture]
        for reversal in (0.0, -0.5, -1.0):
            roll_target = jnp.float32(reversal) * entry_sign
            roll_release = nominal.at[1:, roll_coordinates].set(
                roll_target[None, :]
            )
            combined = roll_release.at[
                1, self.spec.total_width:
            ].add(pitch_residual)
            candidates.extend([
                roll_release,
                jnp.clip(combined, -1.0, 1.0),
            ])
        return jnp.stack(candidates, axis=0)

    def normal_rescue_plans(
        self, state, reference_nodes, horizon_steps, t0=0.0,
    ):
        """Task-owned NORMAL candidates for a failed ordinary replan.

        Fixed-stance displacement tasks first lower the contact point along
        the existing rear-face chart while retaining the incumbent force and
        realized stiffness.  This reduces the external moment without removing
        the ankle brace that balances a high requested load.  Only then try
        monotonically shedding the requested normal load.  Every proposal is
        still accepted solely by the complete model certificate.  The
        force-step task is deliberately excluded: changing its commanded load
        or contact point would invalidate the P1 tracking experiment.

        A certified UNLOAD-to-NORMAL recovery can leave the physical joints a
        few control frames behind the external DIAL reference.  The shifted
        recovery remains safe over its immediate backup, yet its newly
        appended horizon tail can then miss the terminal upright core.  Rather
        than rewinding the task clock or increasing residual authority,
        express one to three frames of reference lag in the existing residual
        chart and add it to the strongest ordered momentum/roll recovery.

        ``horizon_steps`` comes from the solver contract.  Node offsets use
        the same uniform knot grid as DIAL's quadratic spline, so the task does
        not assume a particular Hsample/Hnode pair.  These are proposals only:
        the backend calls this hook after every ordinary path is unsafe and
        may execute only the first complete NORMAL horizon certified by the
        unchanged model-based gate.
        """
        del t0
        if not self._is_walk:
            if self._bcfg.fixed_force_target:
                return None
            node_count = int(reference_nodes.shape[0])
            if node_count < 2:
                return None
            cfg = self._bcfg
            force_span = max(float(cfg.f_max - cfg.f_min), 1.0e-6)
            start_nu = jnp.clip(
                reference_nodes[0, self.spec.nu_slice], -1.0, 1.0
            )
            candidates = []
            # ``b`` is always the final coordinate of the task's position
            # chart (with or without an optional base prefix).  Keep node zero
            # exactly continuous, then lower the rear-face contact target.
            # The ordered mid/low bank lets the certificate choose the least
            # geometry change that restores a safe horizon.
            b_index = self.spec.r_slice.stop - 1
            start_b = jnp.clip(reference_nodes[0, b_index], -1.0, 1.0)
            for target_b in (-0.5, -1.0):
                b_ramp = jnp.linspace(
                    start_b,
                    jnp.float32(target_b),
                    node_count,
                    dtype=reference_nodes.dtype,
                )
                candidates.append(reference_nodes.at[:, b_index].set(b_ramp))
            # Ordered from the smallest intervention that can keep useful
            # pushing authority to a zero-force NORMAL hold.  The first node
            # is exactly the committed action; only future knots are tapered.
            for fraction in (0.875, 0.75, 0.5, 0.0):
                target_force = jnp.clip(
                    jnp.float32(fraction * cfg.f_target),
                    jnp.float32(cfg.f_min),
                    jnp.float32(cfg.f_max),
                )
                target_nu = jnp.clip(
                    2.0 * (target_force - cfg.f_min) / force_span - 1.0,
                    -1.0,
                    1.0,
                )
                force_ramp = jnp.linspace(
                    start_nu.squeeze(-1),
                    target_nu,
                    node_count,
                    dtype=reference_nodes.dtype,
                )
                candidate = reference_nodes.at[:, self.spec.nu_slice].set(
                    force_ramp[:, None]
                )
                candidates.append(candidate)
                # If the safe-set boundary is inside the first knot interval,
                # a slow ramp keeps the unsafe load active too long.  Expose
                # the corresponding immediate load shed as a separate,
                # still fully certified NORMAL candidate.
                candidates.append(
                    reference_nodes.at[:, self.spec.nu_slice].set(target_nu)
                )
            return jnp.stack(candidates, axis=0)
        if not (
            self._is_walk
            and self._bcfg.walk_leg_control == "joint_target"
            and getattr(self, "_walk_joint_reference", None) is not None
            and self._bcfg.emergency_reference_rewind_steps > 0
        ):
            return None
        horizon_steps = int(horizon_steps)
        if horizon_steps < 1:
            raise ValueError("normal rescue requires a positive horizon")
        recovery_bank = self.normal_recovery_plans(
            state, reference_nodes, 0.0
        )
        if recovery_bank is None:
            return None
        base = recovery_bank[-1]
        node_count = int(reference_nodes.shape[0])
        node_offsets = jnp.rint(jnp.linspace(
            0.0, float(horizon_steps), node_count
        )).astype(jnp.int32)
        phase = jnp.asarray(state.info["walk_reference_step"], jnp.int32)
        current = jnp.clip(
            phase + node_offsets,
            jnp.int32(0),
            jnp.int32(self._walk_joint_reference.shape[0] - 1),
        )
        residual_scale = jnp.float32(
            self._bcfg.walk_joint_reference_residual_scale
        )
        if float(residual_scale) <= 0.0:
            return None
        candidates = []
        for lag in (1, 2, 3):
            delayed = jnp.maximum(current - jnp.int32(lag), jnp.int32(0))
            delta = jnp.clip(
                (
                    self._walk_joint_reference[delayed]
                    - self._walk_joint_reference[current]
                ) / residual_scale,
                -1.0,
                1.0,
            )
            candidate = base.at[:, self.spec.total_width:].add(delta)
            candidates.append(jnp.clip(candidate, -1.0, 1.0))
        return jnp.stack(candidates, axis=0)

    def emergency_plan_is_active(self, nodes):
        # Plans may be a single ``(H, A)`` sequence or a task-owned bank with
        # arbitrary leading batch dimensions.  Reduce horizon and feedforward
        # action axes while preserving only candidate-bank axes.
        return jnp.all(
            nodes[..., self.spec.nu_slice] <= -1.0 + 1.0e-4,
            axis=(-2, -1),
        )

    def emergency_plan_should_override(self, state):
        ps = state.pipeline_state
        forces = self._box_contact_forces(ps)
        _, g = self._manifold(
            ps, jnp.zeros((self.action_size,), jnp.float32), state.info
        )
        unsafe = jnp.any(self._safety_margins(ps, forces, g[0]) > 0.0)
        # A coast prediction shapes task progress but is not a safety
        # certificate: unloading a floor-level box can let the walking foot
        # catch it.  Emergency override therefore remains physical-safety
        # only; ordinary task completion is handled by the absorbing success
        # state and model-based candidate acceptance.
        return unsafe

    def safety_index(self, state):
        ps = state.pipeline_state
        forces = self._box_contact_forces(ps)
        _, g = self._manifold(
            ps, jnp.zeros((self.action_size,), jnp.float32), state.info
        )
        return jnp.max(self._safety_margins(ps, forces, g[0]))

    def _walk_task_memory_features(self, ps, info):
        """Minimal event memory and nonperiodic clocks for the strict P4 policy.

        Physical q/qdot and the periodic gait phase do not identify a pending
        valid landing, its reference, or elapsed contact-force ramp.  Defaults
        also support the base reset's observation before task info is installed.
        """
        cfg = self._bcfg
        zero = jnp.zeros((2,), jnp.float32)
        feet_x = ps.site_xpos[self._feet_site_id, 0]
        advance = feet_x - jnp.asarray(info.get("walk_landing_x", feet_x), jnp.float32)
        steps = jnp.asarray(info.get("walk_forward_steps", zero), jnp.float32)
        required = max(int(cfg.walk_min_steps_per_foot), 0)
        completed = (jnp.clip(steps / required, 0.0, 1.0)
                     if required else jnp.ones_like(steps))
        hold = jnp.clip(
            jnp.asarray(info.get("walk_goal_hold_time", 0.0), jnp.float32)
            / max(float(cfg.walk_success_hold_time), 1e-6), 0.0, 1.0,
        )
        startup_cap = max(
            float(cfg.approach_time) + float(cfg.force_ramp_time),
            float(cfg.gait_ramp_time) if cfg.walk_gait_reference == "cpg" else 0.0,
            float(self.dt),
        )
        if cfg.walk_objective_mode == "dial":
            startup_cap = max(startup_cap, float(cfg.walk_velocity_ramp_time))
        step = jnp.asarray(info.get("step", 0), jnp.float32)
        startup = jnp.clip(step * self.dt / startup_cap, 0.0, 1.0)
        contact_step = jnp.asarray(info.get("contact_step", step), jnp.float32)
        contact_ramp = jnp.asarray(info.get("contact_acquired", 0.0), jnp.float32) * jnp.clip(
            (step - contact_step) * self.dt / max(float(cfg.force_ramp_time), 1e-6),
            0.0, 1.0,
        )
        if cfg.walk_force_startup_mode == "synchronized":
            # Expose the actual selected NORMAL startup ramp, not its unsynchronized clock.
            contact_ramp = self._normal_force_startup_scale(info)
        return jnp.concatenate([
            jnp.asarray(info.get("walk_swing_seen", zero), jnp.float32),
            jnp.asarray(info.get("walk_foot_loaded", zero), jnp.float32),
            jnp.asarray(info.get("walk_swing_eligible", zero), jnp.float32),
            advance, completed, jnp.asarray([hold, startup, contact_ramp], jnp.float32),
        ])

    def _get_obs(self, ps, info) -> jax.Array:
        box_x = ps.x.pos[self._box_idx - 1, 0]
        goal = info.get("box_goal_x", box_x)               # parent reset() calls this pre-goal
        box_pos = ps.x.pos[self._box_idx - 1]
        box_yaw = brax_math.quat_to_euler(ps.x.rot[self._box_idx - 1])[2]
        forces = self._box_contact_forces(ps)
        feet = ps.site_xpos[self._feet_site_id][:, :2]
        com = ps.x.pos[self._pelvis_idx - 1, :2]
        level_names = ("push_to_line", "heavy_dr", "unjam", "push_walk")
        level = str(self._bcfg.level).lower()
        level_one_hot = jnp.asarray(
            [float(level == name) for name in level_names], jnp.float32
        )
        task_obs = jnp.concatenate([
            jnp.asarray([
                box_x,
                box_pos[1],
                box_yaw,
                goal,
                goal - box_x,
                self._mu,
                self._bcfg.f_target / max(self._bcfg.f_max, 1.0),
                forces["hand"] / max(self._bcfg.f_max, 1.0),
                forces["wall"] / max(self._bcfg.f_max, 1.0),
                forces["nonhand"] / max(self._bcfg.f_max, 1.0),
                jnp.linalg.norm(com - feet.mean(axis=0))
                / max(self._bcfg.support_radius, 1.0e-6),
                self._corridor_clearance(ps),
                info.get("contact_acquired", 0.0),
                info.get("unjam_released", 0.0),
                info.get("force_int", 0.0) / max(self._bcfg.force_int_max, 1.0),
            ], jnp.float32),
            level_one_hot,
        ])
        obs = jnp.concatenate([ps.qpos, ps.qvel, task_obs])
        if self._is_walk and self._bcfg.walk_leg_control == "joint_target":
            obs = jnp.concatenate([obs, self._walk_phase_features(info)])
            if self._bcfg.walk_success_mode == "locomotion":
                obs = jnp.concatenate([obs, self._walk_task_memory_features(ps, info)])
        return obs


class HumanoidBoxPushDomainEnv:
    """Reset-key randomized family for one shape-compatible H1 schema."""

    def __init__(self, domains):
        self.domains = tuple(domains)
        if not self.domains:
            raise ValueError("HumanoidBoxPushDomainEnv needs at least one domain")
        action_sizes = {int(env.action_size) for env in self.domains}
        observation_sizes = {int(env.observation_size) for env in self.domains}
        if len(action_sizes) != 1 or len(observation_sizes) != 1:
            raise ValueError(
                "all H1 training domains must share action/observation sizes"
            )
        self._action_size = action_sizes.pop()
        self._observation_size = observation_sizes.pop()

    @property
    def action_size(self):
        return self._action_size

    @property
    def observation_size(self):
        return self._observation_size

    @property
    def backend(self):
        return self.domains[0].backend

    @property
    def dt(self):
        return self.domains[0].dt

    @property
    def policy_action_transform(self):
        transforms = [getattr(env, "policy_action_transform", None)
                      for env in self.domains]
        if any(transform != transforms[0] for transform in transforms):
            raise ValueError("all H1 training domains must share policy action transform")
        return transforms[0]

    def reset(self, rng):
        domain_index = jax.random.randint(
            rng, (), 0, len(self.domains), dtype=jnp.int32
        )
        branches = tuple(
            (lambda key, env=env: env.reset(key)) for env in self.domains
        )
        state = jax.lax.switch(domain_index, branches, rng)
        return state.replace(info={
            **state.info, "_rl_domain_index": domain_index,
        })

    def step(self, state, action):
        domain_index = state.info["_rl_domain_index"]
        branches = tuple(
            (lambda s, env=env: env.step(s, action)) for env in self.domains
        )
        return jax.lax.switch(domain_index, branches, state)


class HumanoidBoxPushResidualActionEnv:
    """Train a policy residual while preserving the absolute task action API."""

    def __init__(self, env, action_bias, action_scale):
        self.env = env
        self.action_bias = jnp.asarray(action_bias, dtype=jnp.float32)
        self.action_scale = jnp.asarray(action_scale, dtype=jnp.float32)
        expected = (int(env.action_size),)
        if self.action_bias.shape != expected or self.action_scale.shape != expected:
            raise ValueError(
                "H1 residual action bias/scale must match the task action: "
                f"{self.action_bias.shape}, {self.action_scale.shape} != {expected}"
            )
        if bool(jnp.any(self.action_scale <= 0.0)):
            raise ValueError("H1 residual action_scale entries must be positive")

    @property
    def action_size(self):
        return self.env.action_size

    @property
    def observation_size(self):
        return self.env.observation_size

    @property
    def backend(self):
        return self.env.backend

    @property
    def dt(self):
        return self.env.dt

    def reset(self, rng):
        return self.env.reset(rng)

    def step(self, state, residual_action):
        action = jnp.clip(
            self.action_bias + self.action_scale * jnp.asarray(residual_action),
            -1.0,
            1.0,
        )
        return self.env.step(state, action)


__all__ = [
    "HumanoidBoxPushConfig", "HumanoidBoxPushEnv",
    "HumanoidBoxPushDomainEnv", "HumanoidBoxPushResidualActionEnv",
]
