"""Humanoid contact-manifold box pushing (idea.txt).

Extends the existing ``UnitreeH1PushCrateEnv`` (H1 + a 1-DoF x-slide box, mass 30)
into the MDAC box-push contact-manifold task. The H1 pushes ``box_body`` to a
goal ``x`` while staying balanced and keeping hand–box contact.

Double-support push (fixed feet, fixed contact face):
  * Primitive ``u^hum`` = (joint references r_h, S_body) via
    ``core/control.PrimitiveSpec(pos_dim=n_joints, stiff_dim=3, feed_dim=0)`` →
    ``action_size = n_joints + 6``. ``S_body`` (log-SPD, K=exp S) modulates the
    body PD stiffness (the position-stiffness primitive on the joint impedance).
  * Reward ``= −J_hum`` (eq:humanoid_cost, reduced): box→goal + upright + alive.
  * ``constraint_residual`` exposes the contact manifold on the CLEAN state:
    ``h_hand`` (hand–box face contact, eq:hand_box_contact) and ``g_bal``
    (CoM-over-support balance, eq:humanoid_ineq) — feeding the MDAC soft-
    feasibility (AL) seam. (``h_box`` is trivial: the box is x-slide only;
    fixed-face j,a,b and stance-foot switching are follow-ups.)

The box-ground manifold ``h_box`` and the discrete hand-face ``(j,a,b)`` / foot
switching of the full eq:humanoid_manifold are follow-ups — this initial
version keeps the contact face FIXED, so no relaxed one-hot is needed.

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
    BRAX_AVAILABLE = True
except Exception:  # pragma: no cover
    BRAX_AVAILABLE = False
    State = object

from genedynamics.core.control.stiffness import PrimitiveSpec, stiffness_log_to_pd
from genedynamics.envs.domains.humanoid.h1_brax import (
    UnitreeH1PushCrateEnv, UnitreeH1PushCrateEnvConfig,
)


@dataclass
class HumanoidBoxPushConfig(UnitreeH1PushCrateEnvConfig):
    push_dist: float = 0.4          # goal = initial box x + push_dist
    box_half: float = 0.6           # box half-size (static_box size 0.6)
    support_radius: float = 0.25    # balance: CoM xy within this of feet center
    s_ref_diag: float = 0.0         # log body-stiffness ref (kp_scale ref = 1)
    s_scale: float = 0.7            # normalized svec -> log-stiffness range
    # stiffness chart: "log_spd" kp=trace(exp S) | "euclid" softplus | "fixed"/"none" kp=1
    stiffness_mode: str = "log_spd"
    # reward weights
    w_box: float = 5.0
    w_upright: float = 1.0
    w_alive: float = 0.1
    w_contact: float = 0.5


class HumanoidBoxPushEnv(UnitreeH1PushCrateEnv):
    """H1 box pushing with the MDAC position-stiffness primitive + contact manifold."""

    def __init__(self, config: HumanoidBoxPushConfig | None = None):
        cfg = config or HumanoidBoxPushConfig()
        super().__init__(cfg)
        self._bcfg = cfg
        mj = self.sys.mj_model
        bid = lambda n: mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY.value, n)
        self._box_idx = bid("box_body")
        self._l_elbow_idx = bid("left_elbow_link")
        self._r_elbow_idx = bid("right_elbow_link")
        self.spec = PrimitiveSpec(pos_dim=int(self.joint_range.shape[0]), stiff_dim=3, feed_dim=0)
        diag_idx = jnp.cumsum(jnp.arange(3, 0, -1)) - jnp.arange(3, 0, -1)   # svec diag (3x3): 0,3,5
        self._s_ref = jnp.zeros((self.spec.stiff_width,), jnp.float32).at[diag_idx].set(cfg.s_ref_diag)

    @property
    def action_size(self) -> int:
        return self.spec.total_width                       # n_joints + 6

    # --- normalized primitive -> (joint refs, body kp scale) ---
    def _unpack(self, action):
        pos = action[self.spec.r_slice]                    # joint references [-1,1]
        s_raw = action[self.spec.s_slice]                  # body log-stiffness svec [-1,1]
        return pos, self._kp_scale(s_raw)

    def _kp_scale(self, s_raw):
        """Flag-driven body stiffness scale (supports the stiffness ablations)."""
        mode = self._bcfg.stiffness_mode
        if mode in ("fixed", "none"):
            return jnp.float32(1.0)                         # no stiffness primitive
        S_vec = self._s_ref + self._bcfg.s_scale * s_raw
        if mode == "euclid":                               # Euclidean (not log-Euclidean)
            from genedynamics.core.control.stiffness import svec2sym
            k = jnp.mean(jax.nn.softplus(jnp.diagonal(svec2sym(S_vec, 3))))
            return jnp.clip(k, 0.25, 4.0)
        K = stiffness_log_to_pd(S_vec, 3)                  # log_spd: K = exp(S)
        return jnp.clip(jnp.trace(K) / 3.0, 0.25, 4.0)

    # --- MDAC geometry/retraction: clean-state stiffness manifold (no mjx) ---
    # The constant-stiffness manifold (keep log-K near reference) — a function of
    # the PRIMITIVE only. Geometry projects onto its tangent; CFS retraction onto
    # {C=0}. Balance/contact (needs the mjx rollout) is handled by the AL.
    def _mdac_res_node(self, u):                           # u: (nu,) node primitive
        s0, s1 = self.spec.s_slice.start, self.spec.s_slice.stop
        return self._bcfg.s_scale * u[s0:s1]               # log-K deviation from ref

    def mdac_constraint(self, state, Ybar_nodes):
        """Clean-state constraint vector C(U) (the CFS retraction target manifold)."""
        return jax.vmap(self._mdac_res_node)(Ybar_nodes).reshape(-1)

    def mdac_geometry_fn(self, state, Ybar_nodes, t0):
        """Per-node constraint-tangent proxy vectors a_geom (Hnode+1, nu)."""
        return jax.vmap(jax.grad(lambda u: 0.5 * jnp.sum(self._mdac_res_node(u) ** 2)))(Ybar_nodes)

    def _scaled_pd(self, pos, kp_scale, ps):
        joint_target = self.act2joint(pos)
        q = ps.qpos[7:][: joint_target.shape[0]]
        qd = ps.qvel[6:][: joint_target.shape[0]]
        tau = kp_scale * self._config.kp * (joint_target - q) - self._config.kd * qd
        return jnp.clip(tau, self.joint_torque_range[:, 0], self.joint_torque_range[:, 1])

    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)
        box_x0 = state.pipeline_state.x.pos[self._box_idx - 1, 0]
        info = dict(state.info)
        info["box_goal_x"] = box_x0 + self._bcfg.push_dist
        info["box_x0"] = box_x0
        return state.replace(info=info)

    def step(self, state: State, action: jax.Array) -> State:
        pos, kp_scale = self._unpack(action)
        ctrl = self._scaled_pd(pos, kp_scale, state.pipeline_state)
        ps = self.pipeline_step(state.pipeline_state, ctrl)

        reward, done = self._reward_done(ps, state.info)
        info = dict(state.info)
        info["step"] = state.info["step"] + 1
        obs = self._get_obs(ps, info)
        return state.replace(pipeline_state=ps, obs=obs, reward=reward, done=done, info=info)

    # --- box-push reward (= -J_hum reduced) ---
    def _reward_done(self, ps, info):
        cfg = self._bcfg
        x = ps.x
        box_x = x.pos[self._box_idx - 1, 0]
        r_box = -((box_x - info["box_goal_x"]) ** 2)
        vec = brax_math.rotate(jnp.array([0.0, 0.0, 1.0]), x.rot[self._torso_idx - 1])
        r_upright = -jnp.sum((vec - jnp.array([0.0, 0.0, 1.0])) ** 2)
        h, g = self._manifold(ps, info)
        r_contact = -jnp.sum(h ** 2)
        torso_z = x.pos[self._torso_idx - 1, 2]
        done = (jnp.dot(vec, jnp.array([0.0, 0.0, 1.0])) < 0) | (torso_z < 0.5)
        done = done.astype(jnp.float32)
        r_alive = 1.0 - done
        reward = (cfg.w_box * r_box + cfg.w_upright * r_upright
                  + cfg.w_alive * r_alive + cfg.w_contact * r_contact)
        return reward, done

    # --- contact manifold (clean state): h_hand (contact), g_bal (balance) ---
    def _manifold(self, ps, info):
        x = ps.x
        box_pos = x.pos[self._box_idx - 1]
        near_face = box_pos + jnp.array([-self._bcfg.box_half, 0.0, 0.0])   # -x face the H1 pushes
        mean_elbow = 0.5 * (x.pos[self._l_elbow_idx - 1] + x.pos[self._r_elbow_idx - 1])
        h_hand = mean_elbow - near_face                    # hand–box face contact (eq:hand_box_contact)
        com_xy = x.pos[self._pelvis_idx - 1, :2]           # CoM proxy = pelvis xy
        feet_c = ps.site_xpos[self._feet_site_id].mean(axis=0)[:2]
        g_bal = jnp.linalg.norm(com_xy - feet_c) - self._bcfg.support_radius   # balance (eq:humanoid_ineq)
        return h_hand, jnp.array([g_bal])

    def constraint_residual(self, state, action, ctx=None):
        return self._manifold(state.pipeline_state, state.info)
