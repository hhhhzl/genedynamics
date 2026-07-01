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
from genedynamics.envs.domains.humanoid.h1_brax import (
    UnitreeH1PushCrateEnv, UnitreeH1PushCrateEnvConfig,
)
from genedynamics.envs.legged_brax_base import get_foot_step, global_to_body_velocity

_STIFF_D = 3                             # 3×3 Cartesian hand stiffness (task geometry, robot-independent)


@dataclass(frozen=True)
class _RobotSpec:
    """Per-robot joint layout for the whole-body box-push controller. The leading ``n_planner``
    joints (legs + waist/torso) are PLANNER-controlled for balance; both arms run the two-hand
    Cartesian contact impedance. Lets the SAME env logic drive H1 or G1 (``config.robot``)."""
    n_joints: int           # actuated joints
    n_planner: int          # leading joints (legs + waist/torso) the planner controls for balance
    right_arm: tuple        # right-arm joint indices (right-hand Cartesian impedance)
    left_arm: tuple         # left-arm joint indices (left-hand Cartesian impedance)
    elbows: tuple           # (left_elbow, right_elbow) joint indices — straightened for an extended push
    sag_legs: tuple         # ((L hip_pitch, knee, ankle_pitch), (R ...)) — sagittal joints for the L2 gait
    rhand_body: str         # right-hand contactor body
    lhand_body: str         # left-hand contactor body
    scene: str              # box-push scene file (under envs/assets/<robot_dir>/)
    robot_dir: str          # assets sub-directory


_ROBOTS = {
    # H1: 19 joints — legs(10)+torso(1) planner; arms 4-DoF (shoulder p/r/y + elbow). Contactor =
    # elbow_link (H1 has no separate hand body). Leg order: hip_yaw,roll,pitch,knee,ankle.
    "h1": _RobotSpec(19, 11, (15, 16, 17, 18), (11, 12, 13, 14), (13, 17),
                     ((2, 3, 4), (7, 8, 9)),
                     "right_elbow_link", "left_elbow_link",
                     "mjx_scene_h1_box_push.xml", "unitree_h1"),
    # G1: 29 joints — legs(12)+waist(3) planner; arms 7-DoF (shoulder p/r/y + elbow + wrist r/p/y).
    # Contactor = wrist_yaw_link. Leg order: hip_pitch,roll,yaw,knee,ankle_pitch,ankle_roll.
    "g1": _RobotSpec(29, 15, tuple(range(22, 29)), tuple(range(15, 22)), (18, 25),
                     ((0, 3, 4), (6, 9, 10)),
                     "right_wrist_yaw_link", "left_wrist_yaw_link",
                     "mjx_scene_g1_box_push.xml", "unitree_g1"),
}

# G1 joint-space PD gains — ALIGNED TO THE MENAGERIE G1 (kp 75 / ankle-wrist 20, kv 2). The H1
# gains (kp 200, kd 5) are far too stiff + underdamped for the lighter G1 -> ~100 rad/s leg tremor.
# Order: legs 0..11 (hip p/r/y, knee, ankle p/r), waist 12..14, arms 15..28 (shoulder p/r/y, elbow,
# wrist r/p/y) — arms are overridden by the hand impedance so their gains are nominal.
_G1_KP = jnp.asarray(
    [75., 75., 75., 75., 20., 20.] * 2 + [75., 75., 75.] + [75., 75., 75., 75., 20., 20., 20.] * 2, jnp.float32)
# kd = 2 (menagerie kv): the control runs at 50 Hz (one tau per env step, held over 5 physics
# substeps), so a HIGH kd overshoots under zero-order-hold and DEstabilizes (kd 10 -> instant
# blow-up). Trembling is instead curbed by a joint-velocity reward penalty (w_smooth) that makes
# MDAC prefer calm motions, + a gentle leg_scale.
_G1_KD = jnp.asarray([2.] * 29, jnp.float32)


@dataclass
class HumanoidBoxPushConfig(UnitreeH1PushCrateEnvConfig):
    robot: str = "h1"               # "h1" | "g1" — selects the robot layout (_ROBOTS) + scene
    level: str = "push_to_line"     # push_to_line(L1) | heavy_dr | unjam(L3)  (push_walk L2 = later)
    use_base: bool = False          # H4-B: +v_base -> 15D primitive
    dr_seed: int = 0                # H2 domain-randomization draw
    push_dist: float = 0.15         # goal = initial box x + push_dist (short fixed-stance HAND push; L2 walks far)
    box_half: float = 0.55          # crate half-extent (~DIAL-size; top reaches the H1 hand height)
    box_mass: float = 8.0           # crate mass kg — light enough the right arm pushes it (DR varies)
    # HAND push: the contact point sits on the crate's upper face at the H1 hand (elbow_link) reach
    # height ~1.05 m (near shoulder) so the SHORT high H1 arm can meet it WITHOUT leaning the body.
    hand_push_height: float = 1.05
    contact_band: float = 0.12
    box_geom_friction: float = 0.6  # hand-box surface (geom) friction
    box_shape: str = "box"          # box geom type (cylinder/sphere = geom-type DR, later)
    goal_eps: float = 0.1           # success tolerance on box x reaching the line
    # planar-box slide frictionloss (x = push axis; y/yaw = drift resistance keeping the
    # straight-push levels aligned), aligned to the ~50N the H1 arm delivers.
    box_frictionloss: float = 4.0   # push_to_line: box slides freely along x under the push
    unjam_frictionloss: float = 38.0  # unjam: jammed straight, needs rotation/oblique push
    y_frictionloss: float = 20.0    # box y-slide resistance (lateral drift)
    yaw_frictionloss: float = 8.0   # box yaw resistance (rotation; L3 lowers it to allow turning)
    support_radius: float = 0.25    # balance: CoM xy within this of feet center
    hand_offset: float = 0.15       # TWO-hand push: left/right hands contact the face at +-this in y
    # box domain randomization (mass / size / friction) — heavy_dr draws these
    h2_mass_range: tuple = (5.0, 30.0)
    h2_size_range: tuple = (0.3, 0.55)
    # hand stiffness (3×3 log-SPD): K_hand = exp(s_ref_diag) ≈ 55 N/m — stiff enough the arm
    # EXTENDS to the box face against gravity WITHOUT leaning the body (a soft hand can only reach
    # by leaning forward, which tips the H1). The planner modulates S_hand around this.
    s_ref_diag: float = 4.0
    s_scale: float = 1.0
    stiffness_mode: str = "log_spd"     # log_spd | euclid | fixed | none
    # hand impedance / push force
    d_damp: float = 5.0
    f_min: float = 0.0
    f_max: float = 200.0
    f_target: float = 100.0
    mu_hand: float = 0.6            # hand-box Coulomb friction (g_fric)
    v_base_scale: float = 0.15      # H4-B base-lean command scale
    # H2 domain randomization ranges
    h2_friction_range: tuple = (0.3, 1.0)
    h2_boxfric_range: tuple = (15.0, 40.0)   # all unjam-able by the ~50N arm push
    # planner whole-body balance (option c): the legs+torso (joints 0..10) are planner-controlled
    # via residual joint targets around the default pose; the reward (upright/height/alive) makes
    # the planner FIND a balanced push (a static stance can't balance -- it needs the leg DoF).
    leg_scale: float = 0.6          # planner leg-target residual scale (rad per unit action)
    # reward weights (eq:humanoid_cost, reduced)
    w_box: float = 5.0
    w_prog: float = 8.0            # drive the box forward (push) — strong, else the planner just balances
    w_upright: float = 2.0          # keep torso vertical (balance)
    w_height: float = 1.5          # keep torso near standing height (don't sink)
    w_alive: float = 1.0           # bonus for not falling
    w_contact: float = 5.0         # drive the hand ONTO the crate face (so the planner reaches + contacts)
    w_bal: float = 0.5
    w_align: float = 0.5            # straight-push alignment (box y + yaw -> 0; L3 frees yaw)
    w_smooth: float = 0.0          # penalize joint velocity (anti-tremor); G1-only (H1 default 0)
    # DIAL-faithful body-clean terms (DIAL keeps the torso un-twisted + non-spinning so the walk is
    # balanced; without these + with a too-high w_upright the body contorts). DIAL: yaw x0.1, ang_vel x1.
    w_yaw: float = 0.1             # penalize torso YAW (face forward, don't twist)
    w_angvel: float = 0.1          # penalize torso angular velocity (don't spin/wobble)
    # L2 push_walk (DIAL-style gait): the robot STEPS (dynamic balance) while pushing. The gait
    # reward dominates (like DIAL's x5) so the feet track a stepping pattern; vel walks it forward.
    w_gait: float = 0.0            # foot stepping-pattern tracking (L2 only; set in __init__)
    w_vel: float = 0.0             # forward walking velocity (L2 only)
    target_vx: float = 0.25        # L2 forward walk speed (slow push-walk)
    # L1 lean-and-BRACE (human-like): the CoM must stay over the FEET (not topple onto the box), and
    # a staggered (front-foot-forward) stance is rewarded so the support extends forward -> it can
    # lean to push while braced by the rear leg. G1-L1 only (H1 / L2 default 0).
    w_brace: float = 0.0           # penalize CoM x past the front foot (toppling forward)
    w_stagger: float = 0.0         # reward a forward (staggered) stance so the support spans the lean
    # arm posture: bias the whole (redundant 7-DoF) arm toward a clean two-hand PUSH pose — shoulders
    # forward, elbows bent, palms facing the box at chest height (FK-derived) — so it reaches OUT
    # palm-first, not a folded/drooping arm. G1-only (H1 arm_posture_kp default 0 = off).
    arm_posture_kp: float = 0.0


class HumanoidBoxPushEnv(UnitreeH1PushCrateEnv):
    """Humanoid box pushing/unjamming with the MDAC contact-semantic primitive. Drives H1 or G1
    (``config.robot``) through one robot-agnostic whole-body controller (legs/waist = planner WBC
    balance, both hands = Cartesian contact impedance). Reuses the H1 brax base — G1 shares the
    pelvis/torso_link bodies + left_foot/right_foot sites, so only the scene + joint layout differ."""

    def make_system(self, config):
        # planar-box scene (RIGID box, 3-DoF x/y/yaw); per-robot scene (H1 / G1 motor-actuated).
        from brax.io import mjcf
        from genedynamics.envs.legged_brax_base import get_model_path
        rs = _ROBOTS[str(getattr(config, "robot", "h1")).lower()]
        sys = mjcf.load(get_model_path(rs.robot_dir, rs.scene))
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
                       leg_scale=0.8)   # bounded leg authority (same base for both levels)
            if cfg.level.lower() == "push_walk":   # L2: WALK (reward = gait + forward vel)
                _g1.update(w_gait=30.0, w_vel=1.0, w_upright=2.0, w_bal=0.0, box_frictionloss=8.0)
            else:                                  # L1: STAND + lean-brace (reward = brace + upright)
                _g1.update(w_upright=1.5, w_bal=0.05, box_frictionloss=15.0, w_brace=3.0, w_stagger=1.0)
            for _k, _v in _g1.items():
                if getattr(cfg, _k) == getattr(_h1, _k):
                    setattr(cfg, _k, _v)
        self._rs = _ROBOTS[str(cfg.robot).lower()]
        self._is_walk = str(cfg.level).lower() == "push_walk"
        if self._is_walk:   # L2: CPG gait reference (forces stepping; the planner only corrects it)
            from genedynamics.core.control.bipedal_gait import BipedalGait, GaitParams
            _lg, _rg = self._rs.sag_legs
            self._gait_cpg = BipedalGait(_lg, _rg, self._rs.n_joints, GaitParams(
                cadence=1.2, swing_frac=0.45, hip_amp=0.25, knee_amp=0.5,
                capture_gain=1.0, forward_gain=0.4))
        super().__init__(cfg)
        # keep the robot's actuated joints; drop the 3 planar-box dofs (x/y/yaw) at the tail.
        self.physical_joint_range = self.physical_joint_range[:self._rs.n_joints]
        self.joint_range = self.physical_joint_range
        self._bcfg = cfg
        self._face_select = str(cfg.level).lower() == "unjam"   # j active only for unjam
        self._align_yaw = 0.0 if str(cfg.level).lower() == "unjam" else 1.0  # L3 frees box yaw
        mj = self.sys.mj_model
        bid = lambda n: mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY.value, n)
        self._box_idx = bid("box_body")
        self._rhand_body = bid(self._rs.rhand_body)
        self._lhand_body = bid(self._rs.lhand_body)
        self._box_geom = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM.value, "static_box")
        # nominal standing torso height (robot-agnostic): forward-kinematics the home keyframe.
        _d = mujoco.MjData(mj); _d.qpos[:] = mj.keyframe("home").qpos
        mujoco.mj_forward(mj, _d)
        self._torso_z0 = float(_d.xpos[self._torso_idx, 2])

        # primitive: pos = [v_base(3)?, j(3), a, b]; stiff = svec(3×3)=6; feed = F_n
        pos_dim = (3 if cfg.use_base else 0) + 3 + 2
        self.spec = PrimitiveSpec(pos_dim=pos_dim, stiff_dim=_STIFF_D, feed_dim=1)
        diag_idx = jnp.cumsum(jnp.arange(_STIFF_D, 0, -1)) - jnp.arange(_STIFF_D, 0, -1)
        self._s_ref = jnp.zeros((self.spec.stiff_width,), jnp.float32).at[diag_idx].set(cfg.s_ref_diag)

        nj = self._rs.n_joints
        kp, kd = (_G1_KP, _G1_KD) if cfg.robot.lower() == "g1" else (
            jnp.asarray(self._config.kp, jnp.float32), jnp.asarray(self._config.kd, jnp.float32))
        self._defaultN = jnp.asarray(self._default_pose[:nj], jnp.float32)
        self._kpN = kp[:nj]
        self._kdN = kd[:nj]
        self._tau_lim = jnp.asarray(self.joint_torque_range[:, 1], jnp.float32)[:nj]
        self._arm_mask = jnp.zeros((nj,), bool).at[jnp.asarray(self._rs.right_arm)].set(True)
        self._left_mask = jnp.zeros((nj,), bool).at[jnp.asarray(self._rs.left_arm)].set(True)
        self._arm_any = self._arm_mask | self._left_mask
        # leg+waist joint range for the DIAL-style FULL-authority leg control (L2 walk)
        jr = jnp.asarray(self.physical_joint_range, jnp.float32)
        self._leg_lo = jr[:self._rs.n_planner, 0]
        self._leg_hi = jr[:self._rs.n_planner, 1]
        # H1 is robust enough for DIAL's RAW full-range leg authority (DIAL walks the H1); the lighter
        # G1 NaNs under it, so the G1 uses a BOUNDED residual instead.
        self._full_leg = self._bcfg.robot.lower() == "h1"
        # clean two-hand PUSH pose (FK-derived): shoulders forward (pitch -1.0), elbows bent (1.0),
        # palms facing +x at chest height. The arm posture is biased toward this (G1); H1 uses the
        # default pose (arm_posture_kp 0 -> unused).
        arm_push = self._defaultN
        if cfg.robot.lower() == "g1":
            arm_push = arm_push.at[jnp.asarray(self._rs.right_arm)].set(
                jnp.array([-1.0, -0.2, 0., 1.0, 0., 0., 0.], jnp.float32))
            arm_push = arm_push.at[jnp.asarray(self._rs.left_arm)].set(
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
        self.sys = self.sys.tree_replace({"dof_frictionloss": fl, "body_mass": bm,
                                          "geom_size": gs, "geom_friction": gf, "body_pos": bp})
        self._box_frictionloss = jnp.float32(bf)
        self._mu = jnp.float32(mu)

    @property
    def action_size(self) -> int:
        # hand contact primitive (12) + planner leg+torso targets (n_planner) for WBC balance.
        return self.spec.total_width + self._rs.n_planner

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
        return cfg.f_min + 0.5 * (jnp.clip(nu_raw, -1.0, 1.0) + 1.0) * (cfg.f_max - cfg.f_min)

    def _unpack(self, action):
        cfg = self._bcfg
        pos = action[self.spec.r_slice]
        if cfg.use_base:
            v_base = cfg.v_base_scale * jnp.tanh(pos[:3]); jlog = pos[3:6]; a, b = pos[6], pos[7]
        else:
            v_base = jnp.zeros(3, jnp.float32); jlog = pos[:3]; a, b = pos[3], pos[4]
        # face selection: relaxed one-hot for H4; fixed rear face otherwise
        w_face = jax.nn.softmax(jlog) if self._face_select else jnp.array([1.0, 0.0, 0.0])
        K_hand = self._stiffness(action[self.spec.s_slice])
        F_n = self._force_cmd(action[self.spec.nu_slice][0])
        a01 = 0.5 * (jnp.clip(a, -1.0, 1.0) + 1.0)
        b01 = 0.5 * (jnp.clip(b, -1.0, 1.0) + 1.0)
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
        bz = (cfg.hand_push_height - box_pos[2]) + (2.0 * b - 1.0) * cfg.contact_band
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
    def _one_hand(self, ps, body, p_c, n_c, K_hand, F_n):
        # one hand's Cartesian impedance wrench toward its face contact point p_c.
        p_hand = ps.x.pos[body - 1]
        z_hand = brax_math.rotate(jnp.array([0.0, 0.0, 1.0]), ps.x.rot[body - 1])
        jacp, _ = _mjx_support.jac(self.sys, ps, p_hand, body)         # (nv,3)
        hand_v = jacp.T @ ps.qvel
        wrench = K_hand @ (p_c - p_hand) - self._bcfg.d_damp * hand_v - F_n * n_c
        f_n = jnp.maximum(-jnp.dot(wrench, n_c), 0.0)      # compressive normal into face
        f_t = wrench + f_n * n_c                           # tangential component
        slip = jnp.linalg.norm(hand_v - jnp.dot(hand_v, n_c) * n_c)   # tangential hand speed
        return dict(p_c=p_c, p_hand=p_hand, z_hand=z_hand, jacp=jacp, wrench=wrench,
                    f_n=f_n, f_t=f_t, slip=slip)

    def _hand_contact(self, ps, action):
        # TWO-hand push: both hands run the Cartesian contact impedance toward symmetric points on
        # the selected face (right -> -y, left -> +y in the box frame). Top-level keys = the RIGHT
        # hand (the manifold / metrics read these); the 'left' dict drives the left arm in _control.
        v_base, w_face, a, b, K_hand, F_n = self._unpack(action)
        box_pos = ps.x.pos[self._box_idx - 1]
        box_quat = ps.x.rot[self._box_idx - 1]
        p_c, n_c = self._contact_target(box_pos, box_quat, w_face, a, b)
        off = self._bcfg.hand_offset * brax_math.rotate(jnp.array([0.0, 1.0, 0.0]), box_quat)
        right = self._one_hand(ps, self._rhand_body, p_c - off, n_c, K_hand, F_n)
        left = self._one_hand(ps, self._lhand_body, p_c + off, n_c, K_hand, F_n)
        return dict(v_base=v_base, n_c=n_c, F_n=F_n, left=left, **right)

    def _box_contact_force(self, ps):
        """REAL hand/arm -> box push force: sum of |normal| over active mjx contacts
        that involve the box geom (the physical force actually moving the box, not the
        commanded impedance f_n)."""
        c = ps.contact
        on_box = (c.geom[:, 0] == self._box_geom) | (c.geom[:, 1] == self._box_geom)
        n = c.dist.shape[0]
        fn = jnp.array([_mjx_support.contact_force(self.sys, ps, i)[0] for i in range(n)])
        return jnp.sum(jnp.where(on_box & (c.dist < 0), jnp.abs(fn), 0.0))

    # --- whole-body pi_low: TWO-hand Cartesian contact impedance + PLANNER-controlled legs ---
    # The legs+torso (joints 0..10) follow planner residual targets (the tail of the action) so the
    # planner/MDAC FINDS a balanced push (a static stance can't balance the CoM shift -- it needs the
    # leg DoF, as DIAL does). BOTH arms (11..18) run the hand contact impedance (S_hand primitive).
    def _control(self, ps, action, info):
        c = self._hand_contact(ps, action)                       # reads action[:spec] = the primitive
        nj = self._rs.n_joints
        tau_r = (c["jacp"] @ c["wrench"])[6:6 + nj]              # right-arm Cartesian impedance
        tau_l = (c["left"]["jacp"] @ c["left"]["wrench"])[6:6 + nj]   # left-arm impedance
        q = ps.qpos[7:7 + nj]; qd = ps.qvel[6:6 + nj]
        leg_res = action[self.spec.total_width:]                 # (n_planner,) planner leg+waist targets
        npl = self._rs.n_planner
        # ONE leg base for BOTH levels (DIAL's "planner controls the legs"): the planner sets the
        # leg+waist targets, full-range act2joint for the robust H1 (DIAL's proven locomotion) or
        # bounded around standing for the fragile G1. The REWARD decides: L1 = STAND+brace, L2 = WALK.
        if self._full_leg:   # H1: DIAL raw full-range
            q_tar = self._defaultN.at[:npl].set(
                self._leg_lo + 0.5 * (jnp.clip(leg_res, -1.0, 1.0) + 1.0) * (self._leg_hi - self._leg_lo))
        else:                # G1: bounded residual (full range NaNs it)
            q_tar = self._defaultN.at[:npl].add(self._bcfg.leg_scale * jnp.clip(leg_res, -1.0, 1.0))
        tau_pd = self._kpN * (q_tar - q) - self._kdN * qd        # legs+waist joint-space impedance (PD)
        # arm = hand contact impedance (the S_hand stiffness + F_n push FORCE -> the force-control
        # contact = the MDAC narrative) + a joint posture that places the arm in the clean palm-first
        # push pose (the gross "approach"; the force control lives at the contact). H1 kp=0 -> no-op.
        tau = jnp.where(self._arm_mask, tau_r, tau_pd)
        tau = jnp.where(self._left_mask, tau_l, tau)
        arm_post = self._bcfg.arm_posture_kp * (self._arm_push_pose - q) - 1.0 * qd
        tau = tau + self._arm_any * arm_post
        return jnp.clip(tau, -self._tau_lim, self._tau_lim)

    # --- reset ---
    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)
        box_x0 = state.pipeline_state.x.pos[self._box_idx - 1, 0]
        info = dict(state.info)
        info["box_goal_x"] = box_x0 + self._bcfg.push_dist
        info["box_x0"] = box_x0
        info["step"] = jnp.int32(0)
        return state.replace(info=info)

    def step(self, state: State, action: jax.Array) -> State:
        tau = self._control(state.pipeline_state, action, state.info)
        ps = self.pipeline_step(state.pipeline_state, tau)
        reward, done = self._reward_done(ps, action, state.info)
        info = dict(state.info)
        info["step"] = state.info["step"] + 1
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
        r_prog = box_x - info["box_x0"]                    # progress toward the line
        r_align = -(box_pos[1] ** 2 + (self._align_yaw * box_yaw) ** 2)   # straight push (L3 frees yaw)
        vec = brax_math.rotate(jnp.array([0.0, 0.0, 1.0]), x.rot[self._torso_idx - 1])
        r_upright = -jnp.sum((vec - jnp.array([0.0, 0.0, 1.0])) ** 2)
        h, g = self._manifold(ps, action, info)
        r_contact = -jnp.sum(h[2:5] ** 2)                  # hand-box contact residual (h_hand)
        r_bal = -jnp.maximum(g[0], 0.0) ** 2
        torso_z = x.pos[self._torso_idx - 1, 2]
        r_height = -(torso_z - self._torso_z0) ** 2        # stay near standing height (don't sink)
        done = ((jnp.dot(vec, jnp.array([0.0, 0.0, 1.0])) < 0)
                | (torso_z < 0.6 * self._torso_z0)).astype(jnp.float32)   # fell over / collapsed
        r_alive = 1.0 - done
        r_smooth = -jnp.sum(ps.qvel[6:6 + self._rs.n_joints] ** 2)   # anti-tremor: prefer calm joints
        r_yaw = -brax_math.quat_to_euler(x.rot[self._torso_idx - 1])[2] ** 2   # torso faces fwd (DIAL)
        r_angvel = -jnp.sum(ps.xd.ang[self._torso_idx - 1] ** 2)              # torso not spinning (DIAL)
        reward = (cfg.w_box * r_box + cfg.w_prog * r_prog + cfg.w_upright * r_upright
                  + cfg.w_height * r_height + cfg.w_alive * r_alive + cfg.w_contact * r_contact
                  + cfg.w_bal * r_bal + cfg.w_align * r_align + cfg.w_smooth * r_smooth
                  + cfg.w_yaw * r_yaw + cfg.w_angvel * r_angvel)
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
        # NaN-guard: a physics blow-up in an MDAC rollout must score very low (rejected), not poison
        # the sample-weighted average (-> NaN action). H1 reward is always finite, so this is a no-op there.
        reward = jnp.nan_to_num(reward, nan=-1e3, posinf=-1e3, neginf=-1e3)
        return reward, done

    # --- full contact manifold (eq:humanoid_manifold) ---
    def _manifold(self, ps, action, info):
        cfg = self._bcfg
        x = ps.x
        c = self._hand_contact(ps, action)
        box_pos = x.pos[self._box_idx - 1]
        box_yaw = brax_math.quat_to_euler(x.rot[self._box_idx - 1])[2]
        # planar box: z + roll/pitch are fixed by the planar joint; the box equality is
        # STRAIGHT-PUSH ALIGNMENT — keep lateral y and yaw ~0 (L3 frees the yaw term).
        h_box = jnp.array([box_pos[1], self._align_yaw * box_yaw])     # box_y, box_yaw alignment
        h_hand = c["p_hand"] - c["p_c"]                    # hand on selected face (eq:hand_box_contact)
        h_hand_R = c["z_hand"] + c["n_c"]                  # hand z to -face normal (eq:hand_box_orientation)
        feet = ps.site_xpos[self._feet_site_id]
        h_foot = feet[:, 2]                                # both feet grounded (reduced eq:foot_contact)
        h = jnp.concatenate([h_box, h_hand, h_hand_R, h_foot])     # 2+3+3+2 = 10
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
    def _manifold_res_node(self, u):
        s0, s1 = self.spec.s_slice.start, self.spec.s_slice.stop
        f_span = max(self._bcfg.f_max - self._bcfg.f_min, 1e-6)
        dS = self._bcfg.s_scale * u[s0:s1]
        F = self._force_cmd(u[self.spec.nu_slice][0])
        return jnp.concatenate([dS, jnp.array([(F - self._bcfg.f_target) / f_span])])

    def manifold_residual(self, state, Ybar_nodes):
        return jax.vmap(self._manifold_res_node)(Ybar_nodes).reshape(-1)

    def manifold_geometry(self, state, Ybar_nodes, t0):
        sq = lambda u: 0.5 * jnp.sum(self._manifold_res_node(u) ** 2)
        return jax.vmap(jax.grad(sq))(Ybar_nodes)

    def _get_obs(self, ps, info) -> jax.Array:
        box_x = ps.x.pos[self._box_idx - 1, 0]
        goal = info.get("box_goal_x", box_x)               # parent reset() calls this pre-goal
        return jnp.concatenate([ps.qpos, ps.qvel,
                                jnp.array([box_x, goal, self._mu])])
