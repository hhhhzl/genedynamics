"""Franka-Panda surface-contact scanning env (idea.txt).

A 7-DoF Panda performs contact-rich scanning over a parametric surface. The env
is a brax ``PipelineEnv`` (mjx) over the vendored mesh-free Panda
(``assets/mdac/franka_panda/panda_arm.xml``), consuming the MDAC lower-control
position-stiffness primitive

    u^arm = (Δξ, Δη, Δψ, S_h, F_n^d),   K_h = exp(S_h)   (6×6 spatial stiffness)

(``action_size = 25`` via ``core/control.PrimitiveSpec(3,6,1)``). The in-env
impedance law (π_low, eq 698–724) reconstructs the desired pose from the surface
``p_s(ξ,η)`` / normal ``n_s`` and maps the primitive to joint torques
``τ = Jᵀ F``; the env applies ``τ`` (motor actuators). Reward ``= −J_arm``
(eq:arm_cost). ``constraint_residual`` exposes ``h_surf, h_normal, g_force`` (on
the clean state, no mjx backprop) to the MDAC soft-feasibility (AL) seam.

Normal force uses the quasi-static proxy ``F_n ≈ F_n^d`` (idea.txt; mjx
exposes no contact force). The surfaces are analytic (``core/coverage/surface_geometry``).

Imports guarded so the module loads on fedguide (no brax); construction needs brax.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

try:
    from brax.envs.base import PipelineEnv, State
    from brax.io import mjcf
    from mujoco.mjx._src import support as _mjx_support
    BRAX_AVAILABLE = True
except Exception:  # pragma: no cover
    BRAX_AVAILABLE = False
    PipelineEnv = object
    State = object

from genedynamics.core.control.stiffness import PrimitiveSpec, stiffness_log_to_pd
from genedynamics.core.coverage import surface_geometry as sg

_ASSET = (Path(__file__).resolve().parents[2] / "assets" / "franka_panda" / "panda_arm.xml")
_HOME_QPOS = jnp.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.78], dtype=jnp.float32)


@dataclass
class PandaSurfaceScanConfig:
    dt: float = 0.02
    timestep: float = 0.005
    level: str = "plane"            # surface family (plane / cylinder / ellipsoid)
    standoff: float = 0.0           # d in p_d = p_s + d n_s
    coord_scale: float = 0.05       # (Δξ,Δη,Δψ) step scale
    f_target: float = 5.0           # desired contact normal force
    f_min: float = 0.0
    f_max: float = 20.0
    d_damp: float = 5.0             # impedance damping scale (D = d_damp * I_6)
    s_ref_diag: float = 2.0         # log-stiffness reference (K_ref ≈ exp(2)·I)
    s_scale: float = 3.0            # normalized svec [-1,1] -> log-stiffness offset range
    # stiffness chart: "log_spd" K=exp(S) | "euclid" diag softplus | "fixed" K_ref | "none" I
    stiffness_mode: str = "log_spd"
    # reward weights (eq:arm_cost)
    w_path: float = 5.0
    w_F: float = 0.1
    w_R: float = 1.0
    w_K: float = 0.01
    w_dK: float = 0.01


class PandaSurfaceScanEnv(PipelineEnv):
    """7-DoF Panda surface-contact scanning (MDAC primitive, 6D impedance)."""

    def __init__(self, config: PandaSurfaceScanConfig | None = None, **kw):
        if not BRAX_AVAILABLE:
            raise ImportError("brax + mujoco(mjx) required for PandaSurfaceScanEnv.")
        cfg = config or PandaSurfaceScanConfig(**kw)
        self._config = cfg
        sys = mjcf.load(str(_ASSET))
        n_frames = max(1, int(round(cfg.dt / cfg.timestep)))
        super().__init__(sys=sys, backend="mjx", n_frames=n_frames)

        self.spec = PrimitiveSpec(pos_dim=3, stiff_dim=6, feed_dim=1)   # -> 25
        self.surface = sg.surface_for_level(cfg.level)
        # EE site/body ids resolved from the (brax-fused) System (link0 fused away)
        self._ee_site = 0                                  # only one site ('ee')
        self._ee_body = int(self.sys.site_bodyid[self._ee_site])
        # log-stiffness reference: diagonal `s_ref_diag` on the svec diagonal
        # positions (upper-tri row-major: 0,6,11,15,18,20 for d=6).
        diag_idx = jnp.cumsum(jnp.arange(6, 0, -1)) - jnp.arange(6, 0, -1)
        self._s_ref = jnp.zeros((self.spec.stiff_width,), jnp.float32).at[diag_idx].set(cfg.s_ref_diag)

    # --- action is the MDAC primitive, not the 7 joint torques ---
    @property
    def action_size(self) -> int:
        return self.spec.total_width                       # 25

    # --- reset ---
    def reset(self, rng: jax.Array) -> State:
        ps = self.pipeline_init(_HOME_QPOS, jnp.zeros(self.sys.qd_size()))
        info = {
            "xi": jnp.float32(0.1), "eta": jnp.float32(0.5), "psi": jnp.float32(0.0),
            "prev_s": self._s_ref, "step": jnp.int32(0),
        }
        obs = self._get_obs(ps, info)
        return State(ps, obs, jnp.float32(0.0), jnp.float32(0.0),
                     {"reward": jnp.float32(0.0)}, info)

    # --- surface desired pose + EE kinematics ---
    def _desired_pose(self, xi, eta):
        p_s = sg.point(self.surface, xi, eta)
        n_s = sg.normal(self.surface, xi, eta)
        p_d = p_s + self._config.standoff * n_s
        z_d = -n_s
        t_xi, _ = sg.tangents(self.surface, xi, eta)
        x_d = t_xi / (jnp.linalg.norm(t_xi) + 1e-9)
        y_d = jnp.cross(z_d, x_d); y_d = y_d / (jnp.linalg.norm(y_d) + 1e-9)
        x_d = jnp.cross(y_d, z_d)
        R_d = jnp.stack([x_d, y_d, z_d], axis=1)
        return p_s, n_s, p_d, R_d

    def _ee_kin(self, ps):
        p_h = ps.site_xpos[self._ee_site]
        R_h = ps.site_xmat[self._ee_site].reshape(3, 3)
        jacp, jacr = _mjx_support.jac(self.sys, ps, p_h, self._ee_body)   # (nv,3) each
        lin_v = jacp.T @ ps.qvel
        ang_v = jacr.T @ ps.qvel
        return p_h, R_h, jacp, jacr, lin_v, ang_v

    @staticmethod
    def _orient_err(R_h, R_d):
        return 0.5 * (jnp.cross(R_h[:, 0], R_d[:, 0])
                      + jnp.cross(R_h[:, 1], R_d[:, 1])
                      + jnp.cross(R_h[:, 2], R_d[:, 2]))

    # --- stiffness chart (flag-driven; supports the stiffness ablations) ---
    def _stiffness(self, s_vec):
        mode = self._config.stiffness_mode
        if mode == "fixed":
            return stiffness_log_to_pd(self._s_ref, 6)     # K_ref, ignore S block
        if mode == "none":
            return jnp.eye(6, dtype=jnp.float32)           # minimal/identity stiffness
        if mode == "euclid":                               # Euclidean SPD (not log-Euclidean)
            from genedynamics.core.control.stiffness import svec2sym
            d = jax.nn.softplus(jnp.diagonal(svec2sym(s_vec, 6)))
            return jnp.diag(d)
        return stiffness_log_to_pd(s_vec, 6)               # log_spd (default): K = exp(S)

    # --- impedance π_low: primitive -> joint torque (eq 698–724) ---
    def _impedance_tau(self, ps, xi, eta, s_vec, F_n):
        K = self._stiffness(s_vec)                         # 6×6 SPD spatial stiffness
        D = self._config.d_damp * jnp.eye(6, dtype=jnp.float32)
        _, n_s, p_d, R_d = self._desired_pose(xi, eta)
        p_h, R_h, jacp, jacr, lin_v, ang_v = self._ee_kin(ps)
        pose_err = jnp.concatenate([p_d - p_h, self._orient_err(R_h, R_d)])
        vel_err = jnp.concatenate([-lin_v, -ang_v])
        wrench = K @ pose_err + D @ vel_err + jnp.concatenate([F_n * n_s, jnp.zeros(3)])
        tau = jacp @ wrench[:3] + jacr @ wrench[3:]
        lim = jnp.array([87, 87, 87, 87, 12, 12, 12], jnp.float32)
        return jnp.clip(tau, -lim, lim)

    # --- normalized [-1,1] primitive -> physical (force/stiffness) ---
    def _unpack(self, action):
        cfg = self._config
        r = action[self.spec.r_slice]                      # (Δξ,Δη,Δψ) in [-1,1]
        s_raw = action[self.spec.s_slice]                  # svec in [-1,1]
        nu_raw = action[self.spec.nu_slice]                # F_n in [-1,1]
        S_vec = self._s_ref + cfg.s_scale * s_raw          # physical log-stiffness
        F_n = cfg.f_min + 0.5 * (jnp.clip(nu_raw[0], -1.0, 1.0) + 1.0) * (cfg.f_max - cfg.f_min)
        return r, S_vec, F_n

    # --- step ---
    def step(self, state: State, action: jax.Array) -> State:
        cfg = self._config
        r, S_vec, F_n = self._unpack(action)
        xi = jnp.clip(state.info["xi"] + cfg.coord_scale * r[0], 0.0, 1.0)
        eta = jnp.clip(state.info["eta"] + cfg.coord_scale * r[1], 0.0, 1.0)
        psi = state.info["psi"] + cfg.coord_scale * r[2]

        tau = self._impedance_tau(state.pipeline_state, xi, eta, S_vec, F_n)
        ps = self.pipeline_step(state.pipeline_state, tau)

        reward = self._reward(ps, xi, eta, F_n, S_vec, state.info["prev_s"], state.info["step"])
        info = {"xi": xi, "eta": eta, "psi": psi, "prev_s": S_vec,
                "step": state.info["step"] + 1}
        obs = self._get_obs(ps, info)
        return state.replace(pipeline_state=ps, obs=obs, reward=reward,
                             metrics={"reward": reward}, info=info)

    def _target(self, step):
        # straight scan path in (ξ,η): ξ* ramps 0->1, η* fixed mid
        t = jnp.clip(step.astype(jnp.float32) / 40.0, 0.0, 1.0)
        return t, jnp.float32(0.5)

    # --- MDAC geometry/retraction: clean-state constraint in node space ---
    # The analytic path-/force-tracking manifold (surface-coord target + desired
    # normal force) — a function of the PRIMITIVE only (no mjx rollout, no physics
    # backprop). The geometry seam projects the diffusion update onto its tangent;
    # the CFS retraction projects onto {C=0}. The physics-coupled surface
    # attachment is handled by the AL soft-feasibility.
    def _mdac_res_node(self, u, xi0, eta0, xi_t, eta_t):
        cfg = self._config
        f_span = max(cfg.f_max - cfg.f_min, 1e-6)
        r = u[self.spec.r_slice]
        nu_raw = u[self.spec.nu_slice]
        xi = xi0 + cfg.coord_scale * r[0]
        eta = eta0 + cfg.coord_scale * r[1]
        F_n = cfg.f_min + 0.5 * (jnp.clip(nu_raw[0], -1.0, 1.0) + 1.0) * f_span
        return jnp.array([xi - xi_t, eta - eta_t, (F_n - cfg.f_target) / f_span])

    def mdac_constraint(self, state, Ybar_nodes):
        """Clean-state constraint vector C(U) (the CFS retraction target manifold)."""
        xi0, eta0 = state.info["xi"], state.info["eta"]
        xi_t, eta_t = self._target(state.info["step"])
        return jax.vmap(lambda u: self._mdac_res_node(u, xi0, eta0, xi_t, eta_t))(Ybar_nodes).reshape(-1)

    def mdac_geometry_fn(self, state, Ybar_nodes, t0):
        """Per-node constraint-tangent proxy vectors a_geom (Hnode+1, nu)."""
        xi0, eta0 = state.info["xi"], state.info["eta"]
        xi_t, eta_t = self._target(state.info["step"])
        sq = lambda u: 0.5 * jnp.sum(self._mdac_res_node(u, xi0, eta0, xi_t, eta_t) ** 2)
        return jax.vmap(jax.grad(sq))(Ybar_nodes)

    def _reward(self, ps, xi, eta, F_n, s_vec, prev_s, step):
        cfg = self._config
        _, n_s, _, R_d = self._desired_pose(xi, eta)
        R_h = ps.site_xmat[self._ee_site].reshape(3, 3)
        xi_s, eta_s = self._target(step)
        path = (xi - xi_s) ** 2 + (eta - eta_s) ** 2
        force = (F_n - cfg.f_target) ** 2
        normal = jnp.sum((R_h[:, 2] + n_s) ** 2)
        kreg = jnp.sum((s_vec - self._s_ref) ** 2)
        dk = jnp.sum((s_vec - prev_s) ** 2)
        J = cfg.w_path * path + cfg.w_F * force + cfg.w_R * normal + cfg.w_K * kreg + cfg.w_dK * dk
        return -J

    def _get_obs(self, ps, info) -> jax.Array:
        p_h = ps.site_xpos[self._ee_site]
        return jnp.concatenate([ps.qpos, ps.qvel, p_h,
                                jnp.array([info["xi"], info["eta"], info["psi"]])])

    # --- MDAC soft-feasibility hook: h_surf (eq 727) + h_normal (eq 731) ---
    # The force bound g_force is vacuous here (the affine map keeps F_n^d in
    # [f_min,f_max] by construction), so the active manifold is the 6 equalities
    # (reach the surface + align the EE z to -n_s). Evaluated on the clean
    # predicted state (kinematics + analytic surface), never through mjx backprop.
    def constraint_residual(self, state, action, ctx=None):
        ps = state.pipeline_state
        xi, eta = state.info["xi"], state.info["eta"]
        _, n_s, p_d, _ = self._desired_pose(xi, eta)
        p_h = ps.site_xpos[self._ee_site]
        R_h = ps.site_xmat[self._ee_site].reshape(3, 3)
        h_surf = p_h - p_d                                  # on-surface (eq 727)
        h_normal = R_h[:, 2] + n_s                          # z aligned to -n_s (eq 731)
        h = jnp.concatenate([h_surf, h_normal])             # 6 equalities
        g = jnp.zeros((0,), jnp.float32)
        return h, g
