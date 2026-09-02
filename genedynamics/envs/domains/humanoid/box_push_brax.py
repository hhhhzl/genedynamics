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
    gait_ramp_time: float = 0.5    # smoothly engage the CPG after reset
    gait_cadence: float = 0.8
    gait_swing_frac: float = 0.45
    gait_hip_amp: float = 0.15
    gait_knee_amp: float = 0.30
    gait_capture_gain: float = 0.25
    gait_capture_lead: float = 0.10
    gait_forward_gain: float = 0.40
    gait_roll_amp: float = 0.15
    gait_roll_capture_gain: float = 1.5
    gait_roll_capture_lead: float = 0.10
    gait_roll_limit: float = 0.30
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
        if self._is_walk:   # L2: CPG gait reference (forces stepping; the planner only corrects it)
            from genedynamics.core.control.bipedal_gait import BipedalGait, GaitParams
            _lg, _rg = self._sag_legs
            self._gait_cpg = BipedalGait(
                _lg, _rg, self._robot_profile.num_actuated, GaitParams(
                cadence=cfg.gait_cadence,
                swing_frac=cfg.gait_swing_frac,
                hip_amp=cfg.gait_hip_amp,
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
            site_pos = self.sys.site_pos.at[goal_site, 0].set(jnp.float32(box_x0 + cfg.push_dist))
            self.sys = self.sys.tree_replace({"site_pos": site_pos})
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
        )

    @property
    def action_size(self) -> int:
        # Fixed double-support levels expose only the contact-semantic primitive.  Leg targets are
        # part of the later push_walk controller, not an alternate way to strike the box.
        return self.spec.total_width + (self._n_planner if self._is_walk else 0)

    @property
    def manifold_constraint_size(self) -> int:
        """Action-controllable equality dimension used by ATACOM.

        Fixed stance constrains the six stiffness coordinates and force.  P3
        additionally constrains its three face logits and lateral face
        coordinate.  P4 leaves stiffness/contact selection to the whole-body
        policy and retains only the desired-force equality.
        """
        if self._is_walk:
            return 1
        return 11 if self._face_select else 7

    @property
    def inequality_constraint_size(self) -> int:
        return 3

    @property
    def reliability_feature_size(self) -> int:
        # Fixed across P1--P4 despite the different action dimensions.
        return 24

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
        ax = (2.0 * a - 1.0) * h
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

    def _hand_contact(self, ps, action, info=None):
        # TWO-hand push: both hands run the Cartesian contact impedance toward symmetric points on
        # the selected face (right -> -y, left -> +y in the box frame). Top-level keys = the RIGHT
        # hand (the manifold / metrics read these); the 'left' dict drives the left arm in _control.
        v_base, w_face, a, b, K_hand, F_n_cmd = self._unpack(action)
        # Approach is position-impedance only.  Desired feed-forward force ramps
        # after physical contact acquisition, avoiding a reset-time force step.
        if info is None:
            force_scale = jnp.float32(1.0)
        else:
            acquired = jnp.asarray(info.get("contact_acquired", 0.0), jnp.float32)
            contact_step = jnp.asarray(info.get("contact_step", info.get("step", 0)), jnp.float32)
            elapsed = jnp.maximum(jnp.asarray(info.get("step", 0), jnp.float32) - contact_step, 0.0) * self.dt
            force_scale = acquired * jnp.clip(elapsed / max(float(self._bcfg.force_ramp_time), 1e-6), 0.0, 1.0)
        F_n = F_n_cmd * force_scale
        measured = self._box_contact_force(ps)
        force_int = (jnp.float32(0.0) if info is None
                     else jnp.asarray(info.get("force_int", 0.0), jnp.float32))
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
        return dict(v_base=v_base, n_c=n_c, p_surface=p_surface,
                    F_n=F_n, F_n_cmd=F_n_cmd,
                    F_eff=F_eff, force_scale=force_scale,
                    approach_alpha=approach_alpha, left=left, **right)

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
        return {
            "hand": jnp.sum(jnp.where(active & on_hand, jnp.abs(fn), 0.0)),
            "wall": jnp.sum(jnp.where(active & on_wall, jnp.abs(fn), 0.0)),
            "nonhand": jnp.sum(jnp.where(
                active & ~on_hand & ~on_wall & ~on_floor, jnp.abs(fn), 0.0
            )),
        }

    def _box_contact_force(self, ps):
        """Backward-compatible hand-only physical box force."""
        return self._box_contact_forces(ps)["hand"]

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
        info["unjam_released"] = jnp.float32(0.0)
        info["prev_action"] = jnp.zeros((self.action_size,), jnp.float32)
        return state.replace(info=info)

    def step(self, state: State, action: jax.Array) -> State:
        cfg = self._bcfg
        if cfg.fast_force_loop:
            # Same hierarchy as the arm force-control task: the MPC command is held for one
            # 20ms control interval, while impedance, stance PD and measured-force PI are
            # recomputed at every 4ms physics substep.
            def _substep(carry, _):
                ps_i, force_int = carry
                info_i = {**state.info, "force_int": force_int}
                tau_i = self._control(ps_i, action, info_i)
                ps_i = self._pipeline.step(self.sys, ps_i, tau_i, self._debug)
                contact_i = self._hand_contact(ps_i, action, info_i)
                desired = contact_i["F_n"]
                measured_i = self._box_contact_force(ps_i)
                proposed = jnp.clip(
                    force_int + cfg.ki_force * (desired - measured_i),
                    -cfg.force_int_max,
                    cfg.force_int_max,
                )
                regulating = measured_i >= cfg.contact_acquire_force
                # A moving box can leave the fixed-stance arm workspace.  Do not wind the
                # force integrator to its positive limit while the hands are no longer near it.
                force_int = jnp.where(regulating, proposed, 0.9 * force_int)
                return (ps_i, force_int), None

            (ps, force_int), _ = jax.lax.scan(
                _substep,
                (state.pipeline_state, state.info["force_int"]),
                (),
                self._n_frames,
            )
        else:
            tau = self._control(state.pipeline_state, action, state.info)
            ps = self.pipeline_step(state.pipeline_state, tau)
            contact_now = self._hand_contact(ps, action, state.info)
            desired = contact_now["F_n"]
            measured = self._box_contact_force(ps)
            proposed = jnp.clip(
                state.info["force_int"] + cfg.ki_force * (desired - measured),
                -cfg.force_int_max,
                cfg.force_int_max,
            )
            regulating = measured >= cfg.contact_acquire_force
            force_int = jnp.where(regulating, proposed, 0.9 * state.info["force_int"])
        reward, done = self._reward_done(ps, action, state.info)
        info = dict(state.info)
        info["step"] = state.info["step"] + 1
        info["force_int"] = force_int
        measured = self._box_contact_force(ps)
        contact = self._hand_contact(ps, action, state.info)
        gap_r = jnp.dot(contact["p_hand"] - contact["p_c"], contact["n_c"])
        gap_l = jnp.dot(contact["left"]["p_hand"] - contact["left"]["p_c"], contact["n_c"])
        in_approach_band = jnp.maximum(gap_r, gap_l) <= self._bcfg.contact_acquire_gap
        detected = ((measured >= self._bcfg.contact_acquire_force) | in_approach_band)
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
        reached = (jnp.abs(ps.x.pos[self._box_idx - 1, 0] - state.info["box_goal_x"])
                   <= self._bcfg.goal_eps)
        if str(self._bcfg.level).lower() == "unjam":
            box_yaw = brax_math.quat_to_euler(ps.x.rot[self._box_idx - 1])[2]
            reached = reached & (jnp.abs(box_yaw) <= self._bcfg.unjam_yaw_eps)
        info["task_success"] = jnp.maximum(
            state.info["task_success"], reached.astype(jnp.float32))
        info["prev_action"] = jnp.asarray(action, jnp.float32)
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
        reward = jnp.where(terminal, jnp.float32(0.0), reward)
        done = jnp.where(terminal, jnp.float32(1.0), done)
        obs = self._get_obs(ps, info)
        return state.replace(pipeline_state=ps, obs=obs, reward=reward, done=done, info=info)

    # --- reward (= -J_hum reduced) ---
    def _reward_done(self, ps, action, info):
        cfg = self._bcfg
        x = ps.x
        box_pos = x.pos[self._box_idx - 1]
        box_x = box_pos[0]
        box_yaw = brax_math.quat_to_euler(x.rot[self._box_idx - 1])[2]
        r_box = -((box_x - info["box_goal_x"]) ** 2)       # box reaches the goal LINE x
        # Progress saturates at the line.  The previous unbounded linear term
        # paid the controller to keep pushing forever and analytically moved
        # the reward optimum far beyond the requested goal.
        r_prog = jnp.clip(box_x - info["box_x0"], 0.0, cfg.push_dist)
        r_align = -(box_pos[1] ** 2 + (self._align_yaw * box_yaw) ** 2)   # straight push (L3 frees yaw)
        vec = brax_math.rotate(jnp.array([0.0, 0.0, 1.0]), x.rot[self._torso_idx - 1])
        r_upright = -jnp.sum((vec - jnp.array([0.0, 0.0, 1.0])) ** 2)
        h, g = self._manifold(ps, action, info)
        contact = self._hand_contact(ps, action, info)
        box_forces = self._box_contact_forces(ps)
        force_scale = max(float(cfg.f_max), 1e-6)
        # The benchmark owns the desired force.  Scoring realized force against
        # the action-selected command lets a baseline choose zero force and earn
        # a false tracking advantage.  MGA and DIAL therefore share this same
        # time-indexed target ramp in both reward and evaluation.
        elapsed = jnp.asarray(info["step"], jnp.float32) * self.dt
        force_ref_scale = jnp.clip(
            (elapsed - cfg.approach_time) / max(float(cfg.force_ramp_time), 1e-6),
            0.0, 1.0,
        )
        force_ref = jnp.float32(cfg.f_target) * force_ref_scale
        r_force = -((box_forces["hand"] - force_ref) / force_scale) ** 2
        r_force_limit = -(jnp.maximum(box_forces["hand"] - cfg.f_max, 0.0) / force_scale) ** 2
        r_nonhand = -(box_forces["nonhand"] / force_scale) ** 2
        r_contact = -jnp.sum(h[2:5] ** 2)                  # hand-box contact residual (h_hand)
        r_bal = -jnp.maximum(g[0], 0.0) ** 2
        torso_z = x.pos[self._torso_idx - 1, 2]
        r_height = -(torso_z - self._torso_z0) ** 2        # stay near standing height (don't sink)
        fallen = ((jnp.dot(vec, jnp.array([0.0, 0.0, 1.0])) < 0)
                  | (torso_z < 0.6 * self._torso_z0))
        reached = jnp.abs(box_x - info["box_goal_x"]) <= cfg.goal_eps
        if str(cfg.level).lower() == "unjam":
            reached = reached & (jnp.abs(box_yaw) <= cfg.unjam_yaw_eps)
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
            z_feet = ps.site_xpos[self._feet_site_id][:, 2]
            duty, cad, amp = self._gait_params[self._gait]
            z_tar = get_foot_step(duty, cad, amp, self._gait_phase[self._gait], info["step"] * self.dt)
            r_gait = -jnp.sum((z_tar - z_feet) ** 2)
            vb = global_to_body_velocity(ps.xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1])
            r_vel = -(vb[0] - cfg.target_vx) ** 2
            reward = reward + cfg.w_gait * r_gait + cfg.w_vel * r_vel
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
        h = jnp.concatenate([h_box, h_hand, h_hand_R, h_foot])
        com_xy = x.pos[self._pelvis_idx - 1, :2]
        feet_c = feet.mean(axis=0)[:2]
        g_bal = jnp.linalg.norm(com_xy - feet_c) - cfg.support_radius          # balance (eq:humanoid_ineq)
        g_fric = jnp.linalg.norm(c["f_t"]) - self._mu * c["f_n"]               # friction cone
        g_tip = jnp.abs(com_xy[0] - feet_c[0]) - cfg.support_radius            # forward tipping margin
        g = jnp.array([g_bal, g_fric, g_tip])
        return h, g

    def constraint_residual(self, state, action, ctx=None):
        return self._manifold(state.pipeline_state, action, state.info)

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

    def _unjam_contact_a_target(self, state):
        """Receding contact chart: create torque while jammed, center after release.

        The raw unjam chart origin maps to the validated off-centre rear contact.
        Once the physical box has positive corridor clearance, the clean chart
        moves to rear-centre (raw ``-unjam_contact_a_bias``).  This preserves the
        temporal degree of freedom needed for ``unjam -> straight push`` instead
        of retracting every node forever to one static off-centre contact.
        """
        if not self._face_select:
            return jnp.float32(0.0)
        clearance = self._corridor_clearance(state.pipeline_state)
        released = jnp.asarray(
            state.info.get("unjam_released", 0.0), jnp.float32
        ) > 0.5
        return jnp.where(
            released | (clearance >= self._bcfg.unjam_release_clearance),
            -jnp.float32(self._bcfg.unjam_contact_a_bias),
            jnp.float32(0.0),
        )

    def _manifold_res_node(self, u, contact_a_target=0.0):
        s0, s1 = self.spec.s_slice.start, self.spec.s_slice.stop
        f_span = max(self._bcfg.f_max - self._bcfg.f_min, 1e-6)
        dS = self._bcfg.s_scale * u[s0:s1]
        F = self._force_cmd(u[self.spec.nu_slice][0])
        # Fixed-stance P1--P3 use the validated nominal stiffness chart.  P3
        # retains its bounded vertical contact coordinate as a genuine
        # model-based degree of freedom; P4 instead leaves stiffness free for
        # dynamic walk/contact adaptation.
        residuals = []
        if not self._is_walk:
            residuals.append(dS)
        residuals.append(jnp.array([(F - self._bcfg.f_target) / f_span]))
        if self._face_select:
            j0 = 3 if self._bcfg.use_base else 0
            # The raw chart is centered at zero; _unpack maps that origin to
            # the reachable rear/off-centre physical contact while jammed.
            # P3 varies the lateral coordinate to create the release torque;
            # vertical contact remains a bounded rollout/AL decision.
            residuals.append(u[j0:j0 + 3])
            residuals.append(jnp.array([u[j0 + 3] - contact_a_target]))
        return jnp.concatenate(residuals)

    def manifold_residual(self, state, Ybar_nodes):
        contact_a_target = self._unjam_contact_a_target(state)
        return jax.vmap(
            lambda u: self._manifold_res_node(u, contact_a_target)
        )(Ybar_nodes).reshape(-1)

    def manifold_geometry(self, state, Ybar_nodes, t0):
        contact_a_target = self._unjam_contact_a_target(state)
        sq = lambda u: 0.5 * jnp.sum(
            self._manifold_res_node(u, contact_a_target) ** 2
        )
        return jax.vmap(jax.grad(sq))(Ybar_nodes)

    def manifold_residual_horizon(self, state, dense_actions, t0):
        """Task-owned clean contact chart over the complete dense horizon."""
        del t0
        contact_a_target = self._unjam_contact_a_target(state)
        return jax.vmap(
            lambda u: self._manifold_res_node(u, contact_a_target)
        )(dense_actions).reshape(-1)

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
        return {
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
        state_features = jnp.asarray([
            (box[0] - state.info["box_x0"]) / max(cfg.push_dist, 1.0e-6),
            (state.info["box_goal_x"] - box[0]) / max(cfg.push_dist, 1.0e-6),
            box[1] / max(float(self._half), 1.0e-6),
            yaw / 0.1,
            forces["hand"] / max(cfg.f_max, 1.0),
            forces["wall"] / max(cfg.f_max, 1.0),
            forces["nonhand"] / max(cfg.f_max, 1.0),
            jnp.linalg.norm(com - support) / max(cfg.support_radius, 1.0e-6),
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
        cfg = self._bcfg

        def body(s, u):
            s2 = self.step(s, u)
            h, g = self.constraint_residual(s2, u)
            residual = jnp.concatenate([jnp.abs(h), jax.nn.relu(g)])
            penalty = aug_lambda * jnp.sum(residual) + 0.5 * aug_rho * jnp.sum(residual ** 2)
            forces = self._box_contact_forces(s2.pipeline_state)
            up = brax_math.rotate(
                jnp.array([0.0, 0.0, 1.0]),
                s2.pipeline_state.x.rot[self._torso_idx - 1],
            )[2]
            fallen = (
                (up < 0.0)
                | (s2.pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.5)
            ).astype(jnp.float32)
            force_violation = (forces["hand"] > cfg.f_max).astype(jnp.float32)
            invalid = jnp.maximum(
                fallen,
                (forces["nonhand"] > 0.5).astype(jnp.float32),
            )
            balance = jax.nn.relu(g[0]) / max(cfg.support_radius, 1.0e-6)
            force_mae = jnp.abs(forces["hand"] - cfg.f_target) / max(cfg.f_target, 1.0)
            return s2, (s2.reward - penalty, jnp.asarray([
                force_violation, invalid, balance, force_mae,
            ]))

        _, (scores, risks) = jax.lax.scan(body, state, actions)
        tail = max(1, (int(actions.shape[0]) + 4) // 5)
        risk = jnp.asarray([
            jnp.max(risks[:, 0]),
            jnp.max(risks[:, 1]),
            jnp.mean(jnp.sort(risks[:, 2])[-tail:]),
            jnp.mean(risks[:, 3]),
        ])
        return jnp.mean(scores), risk

    def sequence_risk(self, state, actions):
        return self.sequence_score_risk(state, actions)[1]

    def sequence_risk_is_safe(self, risk):
        return (risk[0] <= 1.0e-8) & (risk[1] <= 1.0e-8) & (risk[2] <= 1.0e-8)

    def sequence_risk_is_no_worse(self, candidate, incumbent, tolerance):
        return jnp.all(candidate[:3] <= incumbent[:3] + tolerance[:3])

    def emergency_sequence_score_risk(self, state, actions, aug_lambda=0.0, aug_rho=0.0):
        return self.sequence_score_risk(state, actions[:1], aug_lambda, aug_rho)

    def project_mga_candidate(self, state, nodes):
        del state
        return jnp.clip(nodes, -1.0, 1.0)

    def emergency_plan(self, state, reference_nodes, t0=0.0):
        """Task-owned zero-force unload with nominal rear-face hold."""
        del state, t0
        emergency = jnp.zeros_like(reference_nodes)
        emergency = emergency.at[:, self.spec.nu_slice].set(-1.0)
        # Reduce Cartesian stiffness during unloading.  A later shifted suffix
        # remains zero-force and is revalidated before it can execute.
        emergency = emergency.at[:, self.spec.s_slice].set(-1.0)
        return emergency

    def emergency_plan_is_active(self, nodes):
        return jnp.all(nodes[:, self.spec.nu_slice] <= -1.0 + 1.0e-4)

    def emergency_plan_should_override(self, state):
        ps = state.pipeline_state
        forces = self._box_contact_forces(ps)
        _, g = self._manifold(
            ps, jnp.zeros((self.action_size,), jnp.float32), state.info
        )
        up = brax_math.rotate(
            jnp.array([0.0, 0.0, 1.0]), ps.x.rot[self._torso_idx - 1]
        )[2]
        return (
            (forces["hand"] > self._bcfg.f_max)
            | (forces["nonhand"] > 0.5)
            | (g[0] > 0.0)
            | (up < 0.0)
        )

    def safety_index(self, state):
        ps = state.pipeline_state
        forces = self._box_contact_forces(ps)
        _, g = self._manifold(
            ps, jnp.zeros((self.action_size,), jnp.float32), state.info
        )
        up = brax_math.rotate(
            jnp.array([0.0, 0.0, 1.0]), ps.x.rot[self._torso_idx - 1]
        )[2]
        return jnp.max(jnp.asarray([
            forces["hand"] / max(self._bcfg.f_max, 1.0) - 1.0,
            forces["nonhand"] / max(self._bcfg.f_max, 1.0),
            g[0] / max(self._bcfg.support_radius, 1.0e-6),
            -up,
        ]))

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
        return jnp.concatenate([ps.qpos, ps.qvel, task_obs])


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


__all__ = [
    "HumanoidBoxPushConfig", "HumanoidBoxPushEnv",
    "HumanoidBoxPushDomainEnv",
]
