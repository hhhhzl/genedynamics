"""Humanoid contact-manifold box pushing / unjamming (idea.txt Exp II).

Extends ``UnitreeH1PushCrateEnv`` (H1 + a 1-DoF x-slide box, half-extent 0.6,
high slide frictionloss = "jammed") into the MDAC loco-manipulation task. The
H1 pushes the box to a goal x while keeping hand–box contact, balance, and
friction feasibility.

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
g_tip (ineq), feeding the MDAC soft-feasibility (AL) seam. The clean-state MDAC
geometry proxy is the constant-stiffness + force-target manifold (primitive-only);
the physics-coupled contact / balance go through the AL.

Levels: H1 double-support fixed face/feet; H2 + domain randomization (hand
friction + box slide frictionloss); H4 box-unjamming with relaxed contact-face
selection (j active). H3 walk-and-push (foot primitive) is a later second stage.

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

_N_JOINTS = 19
_RIGHT_ARM = (15, 16, 17, 18)            # right shoulder p/r/y + elbow (joint order)
_STIFF_D = 3                             # 3×3 hand stiffness


@dataclass
class HumanoidBoxPushConfig(UnitreeH1PushCrateEnvConfig):
    level: str = "unjam"            # double_support | heavy_dr | unjam  (walk = later)
    use_base: bool = False          # H4-B: +v_base -> 15D primitive
    dr_seed: int = 0                # H2 domain-randomization draw
    push_dist: float = 0.4          # goal = initial box x + push_dist
    box_half: float = 0.6           # box half-extent (static_box size)
    goal_eps: float = 0.1           # success tolerance on box x
    # box slide frictionloss, aligned to the ~50N the H1 arm can physically deliver:
    # default = easy slide (box clearly moves), unjam = jammed-but-unjam-able.
    box_frictionloss: float = 12.0  # double_support: box slides freely under the push
    unjam_frictionloss: float = 38.0  # unjam: jammed, needs a firm push to break free
    support_radius: float = 0.25    # balance: CoM xy within this of feet center
    # hand stiffness (3×3 log-SPD)
    s_ref_diag: float = 2.0
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
    # reward weights (eq:humanoid_cost, reduced)
    w_box: float = 5.0
    w_prog: float = 1.0
    w_upright: float = 1.0
    w_alive: float = 0.1
    w_contact: float = 0.5
    w_bal: float = 0.5


class HumanoidBoxPushEnv(UnitreeH1PushCrateEnv):
    """H1 box pushing/unjamming with the MDAC contact-semantic primitive."""

    def __init__(self, config: HumanoidBoxPushConfig | None = None, **kw):
        cfg = config or HumanoidBoxPushConfig(**kw)
        super().__init__(cfg)
        self._bcfg = cfg
        self._face_select = str(cfg.level).lower() == "unjam"   # j active only for unjam
        mj = self.sys.mj_model
        bid = lambda n: mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY.value, n)
        self._box_idx = bid("box_body")
        self._rhand_body = bid("right_elbow_link")
        self._lhand_body = bid("left_elbow_link")
        self._box_geom = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM.value, "static_box")

        # primitive: pos = [v_base(3)?, j(3), a, b]; stiff = svec(3×3)=6; feed = F_n
        pos_dim = (3 if cfg.use_base else 0) + 3 + 2
        self.spec = PrimitiveSpec(pos_dim=pos_dim, stiff_dim=_STIFF_D, feed_dim=1)
        diag_idx = jnp.cumsum(jnp.arange(_STIFF_D, 0, -1)) - jnp.arange(_STIFF_D, 0, -1)
        self._s_ref = jnp.zeros((self.spec.stiff_width,), jnp.float32).at[diag_idx].set(cfg.s_ref_diag)

        self._default19 = jnp.asarray(self._default_pose[:_N_JOINTS], jnp.float32)
        self._kp19 = jnp.asarray(self._config.kp, jnp.float32)[:_N_JOINTS]
        self._kd19 = jnp.asarray(self._config.kd, jnp.float32)[:_N_JOINTS]
        self._tau_lim = jnp.asarray(self.joint_torque_range[:, 1], jnp.float32)[:_N_JOINTS]
        self._arm_mask = jnp.zeros((_N_JOINTS,), bool).at[jnp.asarray(_RIGHT_ARM)].set(True)

        # box slide frictionloss per level (the box slide is the last dof), aligned to
        # the ~50N the arm delivers; H2 also randomizes the hand friction mu.
        mu = cfg.mu_hand
        lvl = str(cfg.level).lower()
        if lvl == "heavy_dr":
            rk, rm = jax.random.split(jax.random.PRNGKey(cfg.dr_seed + 7717))
            lo, hi = cfg.h2_friction_range
            mu = float(jax.random.uniform(rk, minval=lo, maxval=hi))
            lo, hi = cfg.h2_boxfric_range
            bf = float(jax.random.uniform(rm, minval=lo, maxval=hi))
        elif lvl == "unjam":
            bf = cfg.unjam_frictionloss
        else:                                                    # double_support
            bf = cfg.box_frictionloss
        fl = self.sys.dof_frictionloss.at[-1].set(bf)
        self.sys = self.sys.tree_replace({"dof_frictionloss": fl})
        self._box_frictionloss = jnp.float32(bf)
        self._mu = jnp.float32(mu)

    @property
    def action_size(self) -> int:
        return self.spec.total_width                       # 12 or 15

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
    def _contact_target(self, box_pos, w_face, a, b):
        h = self._bcfg.box_half
        ax = (2.0 * a - 1.0) * h
        bz = (2.0 * b - 1.0) * h
        rear = box_pos + jnp.array([-h, ax, bz]);   n_rear = jnp.array([-1.0, 0.0, 0.0])
        left = box_pos + jnp.array([ax, h, bz]);    n_left = jnp.array([0.0, 1.0, 0.0])
        right = box_pos + jnp.array([ax, -h, bz]);  n_right = jnp.array([0.0, -1.0, 0.0])
        pts = jnp.stack([rear, left, right]); nrm = jnp.stack([n_rear, n_left, n_right])
        p_c = w_face @ pts
        n_c = w_face @ nrm
        return p_c, n_c / (jnp.linalg.norm(n_c) + 1e-9)

    # --- hand contact wrench (shared by π_low and the friction constraint) ---
    def _hand_contact(self, ps, action):
        v_base, w_face, a, b, K_hand, F_n = self._unpack(action)
        box_pos = ps.x.pos[self._box_idx - 1]
        p_c, n_c = self._contact_target(box_pos, w_face, a, b)
        p_hand = ps.x.pos[self._rhand_body - 1]
        z_hand = brax_math.rotate(jnp.array([0.0, 0.0, 1.0]), ps.x.rot[self._rhand_body - 1])
        jacp, _ = _mjx_support.jac(self.sys, ps, p_hand, self._rhand_body)   # (nv,3)
        hand_v = jacp.T @ ps.qvel
        wrench = K_hand @ (p_c - p_hand) - self._bcfg.d_damp * hand_v - F_n * n_c
        f_n = jnp.maximum(-jnp.dot(wrench, n_c), 0.0)      # compressive normal into face
        f_t = wrench + f_n * n_c                           # tangential component
        slip = jnp.linalg.norm(hand_v - jnp.dot(hand_v, n_c) * n_c)   # tangential hand speed
        return dict(v_base=v_base, p_c=p_c, n_c=n_c, p_hand=p_hand, z_hand=z_hand,
                    jacp=jacp, wrench=wrench, F_n=F_n, f_n=f_n, f_t=f_t, slip=slip)

    def _box_contact_force(self, ps):
        """REAL hand/arm -> box push force: sum of |normal| over active mjx contacts
        that involve the box geom (the physical force actually moving the box, not the
        commanded impedance f_n)."""
        c = ps.contact
        on_box = (c.geom[:, 0] == self._box_geom) | (c.geom[:, 1] == self._box_geom)
        n = c.dist.shape[0]
        fn = jnp.array([_mjx_support.contact_force(self.sys, ps, i)[0] for i in range(n)])
        return jnp.sum(jnp.where(on_box & (c.dist < 0), jnp.abs(fn), 0.0))

    def _base_lean(self, v_base):
        # crude posture lean (fixed feet): forward->torso pitch, lateral->hip roll
        off = jnp.zeros((_N_JOINTS,), jnp.float32)
        off = off.at[10].add(v_base[0])                    # torso lean
        off = off.at[1].add(v_base[1]).at[6].add(v_base[1])  # hip roll lean (both)
        return off

    def _control(self, ps, action):
        c = self._hand_contact(ps, action)
        tau_full = c["jacp"] @ c["wrench"]                 # (nv,)
        tau_hand = tau_full[6:6 + _N_JOINTS]               # arm-dominant joint torques
        q = ps.qpos[7:7 + _N_JOINTS]; qd = ps.qvel[6:6 + _N_JOINTS]
        q_tar = self._default19 + self._base_lean(c["v_base"])
        tau_stance = self._kp19 * (q_tar - q) - self._kd19 * qd
        tau = jnp.where(self._arm_mask, tau_hand, tau_stance)   # arm=impedance, rest=stance
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
        tau = self._control(state.pipeline_state, action)
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
        box_x = x.pos[self._box_idx - 1, 0]
        r_box = -((box_x - info["box_goal_x"]) ** 2)
        r_prog = box_x - info["box_x0"]                    # progress toward goal
        vec = brax_math.rotate(jnp.array([0.0, 0.0, 1.0]), x.rot[self._torso_idx - 1])
        r_upright = -jnp.sum((vec - jnp.array([0.0, 0.0, 1.0])) ** 2)
        h, g = self._manifold(ps, action, info)
        r_contact = -jnp.sum(h[3:6] ** 2)                  # hand-box contact residual
        r_bal = -jnp.maximum(g[0], 0.0) ** 2
        torso_z = x.pos[self._torso_idx - 1, 2]
        done = ((jnp.dot(vec, jnp.array([0.0, 0.0, 1.0])) < 0) | (torso_z < 0.5)).astype(jnp.float32)
        r_alive = 1.0 - done
        reward = (cfg.w_box * r_box + cfg.w_prog * r_prog + cfg.w_upright * r_upright
                  + cfg.w_alive * r_alive + cfg.w_contact * r_contact + cfg.w_bal * r_bal)
        return reward, done

    # --- full contact manifold (eq:humanoid_manifold) ---
    def _manifold(self, ps, action, info):
        cfg = self._bcfg
        x = ps.x
        c = self._hand_contact(ps, action)
        box_pos = x.pos[self._box_idx - 1]
        box_rpy = brax_math.quat_to_euler(x.rot[self._box_idx - 1])
        h_box = jnp.array([box_pos[2] - 0.6, box_rpy[0], box_rpy[1]])   # grounded, no roll/pitch
        h_hand = c["p_hand"] - c["p_c"]                    # hand on selected face (eq:hand_box_contact)
        h_hand_R = c["z_hand"] + c["n_c"]                  # hand z to -face normal (eq:hand_box_orientation)
        feet = ps.site_xpos[self._feet_site_id]
        h_foot = feet[:, 2]                                # both feet grounded (reduced eq:foot_contact)
        h = jnp.concatenate([h_box, h_hand, h_hand_R, h_foot])     # 3+3+3+2 = 11
        com_xy = x.pos[self._pelvis_idx - 1, :2]
        feet_c = feet.mean(axis=0)[:2]
        g_bal = jnp.linalg.norm(com_xy - feet_c) - cfg.support_radius          # balance (eq:humanoid_ineq)
        g_fric = jnp.linalg.norm(c["f_t"]) - self._mu * c["f_n"]               # friction cone
        g_tip = jnp.abs(com_xy[0] - feet_c[0]) - cfg.support_radius            # forward tipping margin
        g = jnp.array([g_bal, g_fric, g_tip])
        return h, g

    def constraint_residual(self, state, action, ctx=None):
        return self._manifold(state.pipeline_state, action, state.info)

    # --- MDAC clean-state geometry proxy (primitive only; no mjx) ---
    # Constant-stiffness + force-target manifold; contact/balance go to the AL.
    def _mdac_res_node(self, u):
        s0, s1 = self.spec.s_slice.start, self.spec.s_slice.stop
        f_span = max(self._bcfg.f_max - self._bcfg.f_min, 1e-6)
        dS = self._bcfg.s_scale * u[s0:s1]
        F = self._force_cmd(u[self.spec.nu_slice][0])
        return jnp.concatenate([dS, jnp.array([(F - self._bcfg.f_target) / f_span])])

    def mdac_constraint(self, state, Ybar_nodes):
        return jax.vmap(self._mdac_res_node)(Ybar_nodes).reshape(-1)

    def mdac_geometry_fn(self, state, Ybar_nodes, t0):
        sq = lambda u: 0.5 * jnp.sum(self._mdac_res_node(u) ** 2)
        return jax.vmap(jax.grad(sq))(Ybar_nodes)

    def _get_obs(self, ps, info) -> jax.Array:
        box_x = ps.x.pos[self._box_idx - 1, 0]
        goal = info.get("box_goal_x", box_x)               # parent reset() calls this pre-goal
        return jnp.concatenate([ps.qpos, ps.qvel,
                                jnp.array([box_x, goal, self._mu])])
