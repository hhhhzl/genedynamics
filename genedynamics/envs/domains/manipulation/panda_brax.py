"""Franka-Panda surface-contact scanning env (idea.txt Exp I).

A 7-DoF Panda performs contact-rich scanning over a parametric surface. The env
is a brax ``PipelineEnv`` (mjx) over the vendored mesh-free Panda
(``assets/franka_panda/panda_arm.xml``), consuming the MDAC lower-control
position-stiffness primitive

    u^arm = (Δξ, Δη, Δψ, S_h, F_n^d),   K_h = exp(S_h) ∈ S³₊₊  (3×3 translational)

``action_size = 10`` via ``core/control.PrimitiveSpec(3, 3, 1)`` (3 surface-coord
increments + svec(3×3)=6 + 1 force). Orientation stiffness is FIXED (``k_orient``,
not in the action) — the probe orientation is determined by the surface normal
``R_d e_z = -n_s``. The 6×6 full task-space stiffness (action 25) is an appendix
scalability ablation, not the main primitive.

The in-env impedance law (π_low, eq 698–724) reconstructs the desired pose from
``p_s(ξ,η)`` / normal ``n_s`` and maps the primitive to joint torques ``τ=Jᵀ F``.
Reward ``= −J_arm`` (eq:arm_cost). ``constraint_residual`` exposes ``h_surf,
h_normal`` (eq) and ``g_force`` (ineq) to the soft-feasibility (AL) seam; the
general ``manifold_residual`` / ``manifold_geometry`` hooks expose the clean-state
path/force manifold for tangent-projection / retraction solvers.

REAL contact (sim-to-real): the analytic surface ``p_s(ξ,η)`` is triangulated into
a mujoco HFIELD geom injected into the Panda model, and the EE probe sphere is made
collidable — so the probe physically contacts the surface (mjx sphere-hfield), with
real reaction force + friction. The normal contact force is read back from
``support.contact_force`` and used for force tracking (no quasi-static proxy). The
impedance leans on the surface (gravity holds light contact); ``f_target`` is the
firm scan press. Surface families (``level`` + ``surface_seed``): plane/cylinder
(analytic), convex / bumpy NURBS, unseen NURBS + domain randomization (contact
friction + surface compliance via the contact solref).

Imports guarded so the module loads on fedguide (no brax); construction needs brax.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

try:
    import mujoco
    from mujoco import mjx
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
_STIFF_D = 3                             # 3×3 translational stiffness (main primitive)
_PROBE_R = 0.02                          # EE probe sphere radius (panda_arm.xml)
_N_DOF = 7                               # actuated joints (fixed-base Panda; nv=7)


def _ee_home_fk():
    """End-effector world position at the home pose (raw mujoco FK; needed to
    place the surface before the contact model is built)."""
    mj0 = mujoco.MjModel.from_xml_path(str(_ASSET))
    d0 = mujoco.MjData(mj0)
    d0.qpos[:7] = np.asarray(_HOME_QPOS)
    mujoco.mj_forward(mj0, d0)
    sid = mujoco.mj_name2id(mj0, mujoco.mjtObj.mjOBJ_SITE.value, "ee")
    return np.asarray(d0.site_xpos[sid])


def _build_contact_model(surface, ee0, n, depth, friction, solref):
    """Place the analytic surface so its scan-start point sits ``depth`` below the
    home EE, triangulate its heights into a mujoco HFIELD, and inject it + a
    collidable EE probe into the Panda model -> a REAL contact model (mjx
    sphere-hfield). Returns (MjModel, placed Surface)."""
    p0 = np.asarray(sg.point(surface, 0.1, 0.5))
    surface = sg.translate(surface, np.array([ee0[0] - p0[0], ee0[1] - p0[1], (ee0[2] - depth) - p0[2]]))
    g = np.linspace(0.0, 1.0, n)
    P = np.array([[np.asarray(sg.point(surface, xi, eta)) for eta in g] for xi in g])  # P[i=xi, j=eta]
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    zlo, zhi = float(Z.min()), float(Z.max()); elev = max(zhi - zlo, 1e-3)
    cx, cy = (float(X.min()) + float(X.max())) / 2, (float(Y.min()) + float(Y.max())) / 2
    rx, ry = (float(X.max()) - float(X.min())) / 2, (float(Y.max()) - float(Y.min())) / 2
    hdata = ((Z - zlo) / elev).T.reshape(-1).astype(np.float32)        # mujoco hfield: [row=y, col=x]

    xml = open(_ASSET).read().replace('contype="0" conaffinity="0" />', 'contype="1" conaffinity="1" />', 1)
    asset = f'<asset><hfield name="surf" nrow="{n}" ncol="{n}" size="{rx:.5f} {ry:.5f} {elev:.5f} 0.05"/></asset>'
    hgeom = (f'<geom name="surf" type="hfield" hfield="surf" pos="{cx:.5f} {cy:.5f} {zlo:.5f}" '
             f'contype="1" conaffinity="1" friction="{friction:.3f} 0.01 0.001" solref="{solref}"/>')
    xml = xml.replace("<worldbody>", asset + "<worldbody>").replace("</worldbody>", hgeom + "</worldbody>")
    mj = mujoco.MjModel.from_xml_string(xml)
    mj.hfield_data[:] = hdata
    return mj, surface


@dataclass
class PandaSurfaceScanConfig:
    dt: float = 0.02
    timestep: float = 0.005
    level: str = "plane"            # surface family: plane/cylinder/s1/s2/s3/s4
    surface_seed: int = 0           # selects the random NURBS (S2-S4) + DR draw
    standoff: float = 0.0           # d in p_d = p_s + d n_s
    coord_scale: float = 0.05       # (Δξ,Δη,Δψ) step scale
    scan_rate: float = 0.006        # target ξ advance/step
    scan_span: float = 0.5          # total ξ sweep of the scan target
    # real mjx contact: surface is an hfield, EE a collidable probe (sphere-hfield)
    hfield_n: int = 36              # hfield grid resolution
    contact_depth: float = 0.025    # surface this far below the home EE (EE starts near contact)
    contact_solref: str = "0.08 1"  # compliant contact: force less sensitive to penetration ->
                                    # lower force variance while tracking curved surfaces (+sim2real)
    f_target: float = 45.0          # desired contact normal force (~the natural soft-contact press)
    track_tol: float = 0.02         # |h_surf| below this = "on the scan path"; force tracking
                                    # is scored only on these steps (a baseline that lags off
                                    # the path gets a trivially steady force that shouldn't count)
    f_min: float = 0.0
    f_max: float = 60.0
    f_cmd_pad: float = 10.0         # commanded force may reach [f_min-pad, f_max+pad]
    d_damp: float = 10.0            # impedance damping scale
    # force tracking = gravity feed-forward (cancels the ~40N weight offset) + PI on the
    # measured-force error. The INTEGRAL (ki_force) is what regulates the force while the
    # EE follows a curved surface (rejects the slowly-varying following disturbance that
    # pure proportional feedback lags on the stiff contact -> cuts the on-curve force
    # variance). force_int is an env state (threaded through info, anti-windup clamped).
    # Robust to unknown surface stiffness (feedback, not model-based) -> sim2real + S4.
    kp_force: float = 0.3
    ki_force: float = 1.0
    force_int_max: float = 30.0     # anti-windup clamp on the force integral
    fast_force_loop: bool = True    # run the impedance/force servo every physics substep
                                    # (real-robot fast force loop under the MPC) vs once/step
    # grav_comp=0: the arm weight is the ROBUST contact-maintenance force on the
    # roughly-horizontal scan surfaces (grav_comp=1 floats off curves). The closed-loop
    # force feedback (kp_force) rejects the weight's contribution so the REAL contact
    # force still tracks the commanded F_n despite the uncompensated weight.
    grav_comp: float = 0.0
    k_orient: float = 50.0          # FIXED orientation stiffness (not in action)
    s_ref_diag: float = 5.3         # log-stiffness reference (K_ref ≈ exp(5.3) ≈ 200 N/m)
    s_scale: float = 2.0            # normalized svec [-1,1] -> log-stiffness offset (exp(3.3..7.3))
    # stiffness chart: "log_spd" K=exp(S) | "euclid" diag softplus | "fixed" K_ref | "none" I
    stiffness_mode: str = "log_spd"
    # domain randomization (S4): surface stiffness + contact friction
    surface_stiffness: float = 1.0e4    # default ~rigid (negligible sink)
    friction: float = 1.0
    s4_stiffness_range: tuple = (2.0e3, 1.0e4)
    s4_friction_range: tuple = (0.3, 1.0)
    # reward weights (eq:arm_cost)
    w_path: float = 5.0
    w_F: float = 1.0
    w_R: float = 1.0
    w_K: float = 0.01
    w_dK: float = 0.01


class PandaSurfaceScanEnv(PipelineEnv):
    """7-DoF Panda surface-contact scanning (MDAC 10D primitive, 3×3 impedance)."""

    def __init__(self, config: PandaSurfaceScanConfig | None = None, **kw):
        if not BRAX_AVAILABLE:
            raise ImportError("brax + mujoco(mjx) required for PandaSurfaceScanEnv.")
        cfg = config or PandaSurfaceScanConfig(**kw)
        self._config = cfg
        self._xi0, self._eta0 = jnp.float32(0.1), jnp.float32(0.5)

        # domain randomization (unseen): contact friction + surface compliance.
        mu, solref = cfg.friction, cfg.contact_solref
        if str(cfg.level).lower() == "unseen":
            rk, rm = jax.random.split(jax.random.PRNGKey(cfg.surface_seed + 9973))
            lo, hi = cfg.s4_friction_range
            mu = float(jax.random.uniform(rm, minval=lo, maxval=hi))
            lo, hi = cfg.s4_stiffness_range            # softer surface -> slower solref time const
            ks = float(jax.random.uniform(rk, minval=lo, maxval=hi))
            solref = f"{0.02 * (1.0e4 / ks):.4f} 1"

        # build the REAL contact model: analytic surface -> mjx hfield, EE -> probe.
        surface = sg.surface_for_level(cfg.level, cfg.surface_seed)
        ee0 = _ee_home_fk()
        mj, self.surface = _build_contact_model(surface, ee0, cfg.hfield_n, cfg.contact_depth, mu, solref)
        sys = mjcf.load_model(mj)
        n_frames = max(1, int(round(cfg.dt / cfg.timestep)))
        super().__init__(sys=sys, backend="mjx", n_frames=n_frames)

        self._mjx_model = mjx.put_model(mj)            # for support.contact_force
        self.spec = PrimitiveSpec(pos_dim=3, stiff_dim=_STIFF_D, feed_dim=1)   # -> 10
        self._ee_site = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE.value, "ee")
        self._ee_body = int(self.sys.site_bodyid[self._ee_site])
        self._probe_geom = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM.value, "probe")
        diag_idx = jnp.cumsum(jnp.arange(_STIFF_D, 0, -1)) - jnp.arange(_STIFF_D, 0, -1)
        self._s_ref = jnp.zeros((self.spec.stiff_width,), jnp.float32).at[diag_idx].set(cfg.s_ref_diag)
        self._mu = jnp.float32(mu)
        self._k_surf = jnp.float32(cfg.surface_stiffness)   # kept for obs/back-compat

    # --- real contact normal force at the EE probe (mjx contact) ---
    def _contact_force(self, ps) -> jnp.ndarray:
        c = ps.contact
        if c is None or c.dist.shape[0] == 0:
            return jnp.float32(0.0)
        n = c.dist.shape[0]
        on_probe = (c.geom[:, 0] == self._probe_geom) | (c.geom[:, 1] == self._probe_geom)
        # support.contact_force indexes efc_address[contact_id] statically -> NOT vmappable,
        # so this stays a python loop over the (small, static) contact buffer.
        fn = jnp.array([_mjx_support.contact_force(self._mjx_model, ps, i)[0] for i in range(n)])
        return jnp.sum(jnp.where(on_probe & (c.dist < 0), jnp.abs(fn), 0.0))

    # --- action is the MDAC primitive, not the 7 joint torques ---
    @property
    def action_size(self) -> int:
        return self.spec.total_width                       # 10

    def reset(self, rng: jax.Array) -> State:
        ps = self.pipeline_init(_HOME_QPOS, jnp.zeros(self.sys.qd_size()))
        info = {
            "xi": self._xi0, "eta": self._eta0, "psi": jnp.float32(0.0),
            "prev_s": self._s_ref, "step": jnp.int32(0),
            "force_int": jnp.float32(0.0),     # contact-force integral (admittance state)
        }
        obs = self._get_obs(ps, info)
        return State(ps, obs, jnp.float32(0.0), jnp.float32(0.0),
                     {"reward": jnp.float32(0.0)}, info)

    # --- surface desired pose + EE kinematics ---
    def _desired_pose(self, xi, eta):
        p_s = sg.point(self.surface, xi, eta)
        n_s = sg.normal(self.surface, xi, eta)
        # orient the outward normal UPWARD (+z): the surfaces are horizontal patches
        # scanned from above, and the Panda tool z-axis points down, so n_s up =>
        # z_d = -n_s = down aligns the down-pointing tool. Family-agnostic.
        n_s = n_s * jnp.sign(n_s[2] + 1e-9)
        # the EE site target rides a probe-radius above the surface (the probe
        # sphere just touches), so the SITE tracks the surface (|h_surf| ~ 0) when
        # the probe is in contact.
        p_d = p_s + (self._config.standoff + _PROBE_R) * n_s
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
        jacp, jacr = _mjx_support.jac(self.sys, ps, p_h, self._ee_body)
        return p_h, R_h, jacp, jacr, jacp.T @ ps.qvel, jacr.T @ ps.qvel

    @staticmethod
    def _orient_err(R_h, R_d):
        return 0.5 * (jnp.cross(R_h[:, 0], R_d[:, 0])
                      + jnp.cross(R_h[:, 1], R_d[:, 1])
                      + jnp.cross(R_h[:, 2], R_d[:, 2]))

    # --- 3×3 translational stiffness chart (flag-driven; stiffness ablations) ---
    def _stiffness(self, s_vec):
        mode = self._config.stiffness_mode
        if mode == "fixed":
            return stiffness_log_to_pd(self._s_ref, _STIFF_D)   # K_ref, ignore S block
        if mode == "none":
            return jnp.eye(_STIFF_D, dtype=jnp.float32)
        if mode == "euclid":                                    # Euclidean SPD (not log)
            from genedynamics.core.control.stiffness import svec2sym
            d = jax.nn.softplus(jnp.diagonal(svec2sym(s_vec, _STIFF_D)))
            return jnp.diag(d)
        return stiffness_log_to_pd(s_vec, _STIFF_D)             # log_spd: K = exp(S)

    # --- impedance π_low: primitive -> joint torque (eq 698–724) ---
    # Translational impedance K_t(S) on position + FIXED k_orient on orientation +
    # an F_n feedforward pressing into the surface. Coulomb friction is now REAL
    # (the mjx probe-hfield contact), so no controller-side tangential cap.
    def _impedance_tau(self, ps, n_s, p_d, R_d, s_vec, F_n, force_int):
        cfg = self._config
        K_t = self._stiffness(s_vec)                            # 3×3 translational SPD
        p_h, R_h, jacp, jacr, lin_v, ang_v = self._ee_kin(ps)
        # FORCE TRACKING: gravity feed-forward + PI on the measured-force error so the REAL
        # contact force tracks the commanded F_n. g_n = arm-weight normal force at the EE
        # (jacp-mapped qfrc_bias on n_s), cancelled so the weight does not bias the press;
        # kp_force = fast proportional; force_int = the INTEGRAL (admittance) state that
        # regulates the slowly-varying following disturbance while tracking a curved
        # surface (the part pure proportional lags on the stiff contact). Weight stays in
        # the dynamics -> stable contact on curves (unlike full grav comp, which floats).
        F_meas = self._contact_force(ps)
        g_ee = jnp.linalg.solve(jacp.T @ jacp + 1e-6 * jnp.eye(3), jacp.T @ ps.qfrc_bias[:_N_DOF])
        g_n = jnp.dot(g_ee, n_s)                              # arm-weight normal force at EE
        F_eff = jnp.clip(F_n - g_n + cfg.kp_force * (F_n - F_meas) + force_int,
                         cfg.f_min - cfg.f_cmd_pad, cfg.f_max + cfg.f_cmd_pad)
        f_pos = K_t @ (p_d - p_h) - cfg.d_damp * lin_v - F_eff * n_s  # press INTO surface (-n_s)
        m_rot = cfg.k_orient * self._orient_err(R_h, R_d) - cfg.d_damp * ang_v
        # PARTIAL gravity/coriolis compensation: full comp makes the arm weightless
        # and it floats off; zero comp makes it lean hard (~20 N). A small residual
        # weight keeps light contact and the planner trims F_n to hit f_target.
        tau = jacp @ f_pos + jacr @ m_rot + cfg.grav_comp * ps.qfrc_bias[: _N_DOF]
        lim = jnp.array([87, 87, 87, 87, 12, 12, 12], jnp.float32)
        return jnp.clip(tau, -lim, lim)

    # --- normalized [-1,1] primitive -> physical ---
    def _force_cmd(self, nu_raw):
        """Commanded normal force; spans [f_min-pad, f_max+pad] so it CAN exceed
        the [f_min,f_max] bound -> g_force is a genuine constraint."""
        cfg = self._config
        lo, hi = cfg.f_min - cfg.f_cmd_pad, cfg.f_max + cfg.f_cmd_pad
        return lo + 0.5 * (jnp.clip(nu_raw, -1.0, 1.0) + 1.0) * (hi - lo)

    def _unpack(self, action):
        cfg = self._config
        r = action[self.spec.r_slice]                      # (Δξ,Δη,Δψ) in [-1,1]
        s_raw = action[self.spec.s_slice]                  # svec(3×3) in [-1,1]
        nu_raw = action[self.spec.nu_slice]                # force in [-1,1]
        S_vec = self._s_ref + cfg.s_scale * s_raw
        return r, S_vec, self._force_cmd(nu_raw[0])

    # --- step ---
    def step(self, state: State, action: jax.Array) -> State:
        cfg = self._config
        r, S_vec, F_n_cmd = self._unpack(action)
        xi = jnp.clip(state.info["xi"] + cfg.coord_scale * r[0], 0.0, 1.0)
        eta = jnp.clip(state.info["eta"] + cfg.coord_scale * r[1], 0.0, 1.0)
        psi = state.info["psi"] + cfg.coord_scale * r[2]
        F_n = jnp.clip(F_n_cmd, cfg.f_min, cfg.f_max)       # applied force stays physical
        # the desired pose (surface point/normal/orientation) is fixed for the control
        # period -> compute it ONCE, not per substep (avoids 4x the NURBS surface eval).
        _, n_s, p_d, R_d = self._desired_pose(xi, eta)

        # FAST INNER FORCE/IMPEDANCE SERVO: recompute the controller (and the force
        # integral) EVERY physics substep instead of holding one torque open-loop for the
        # whole control period. This is the real-robot hierarchy -- a fast (here per-
        # substep, ~200 Hz) force loop under the ~50 Hz MPC -- which regulates the contact
        # force WITHIN the control step (rejecting the following/contact disturbances the
        # MPC-rate loop is too slow for). The MPC command (xi, eta, S_vec, F_n) is held.
        if cfg.fast_force_loop:
            def _substep(carry, _):
                ps_i, fint = carry
                tau_i = self._impedance_tau(ps_i, n_s, p_d, R_d, S_vec, F_n, fint)
                ps_i = self._pipeline.step(self.sys, ps_i, tau_i, self._debug)
                fint = jnp.clip(fint + cfg.ki_force * (F_n - self._contact_force(ps_i)),
                                -cfg.force_int_max, cfg.force_int_max)
                return (ps_i, fint), None
            (ps, force_int), _ = jax.lax.scan(
                _substep, (state.pipeline_state, state.info["force_int"]), (), self._n_frames)
        else:
            force_int = state.info["force_int"]
            tau = self._impedance_tau(state.pipeline_state, n_s, p_d, R_d, S_vec, F_n, force_int)
            ps = self.pipeline_step(state.pipeline_state, tau)
            force_int = jnp.clip(force_int + cfg.ki_force * (F_n - self._contact_force(ps)),
                                 -cfg.force_int_max, cfg.force_int_max)

        reward = self._reward(ps, xi, eta, F_n, S_vec, state.info["prev_s"], state.info["step"])
        info = {"xi": xi, "eta": eta, "psi": psi, "prev_s": S_vec,
                "step": state.info["step"] + 1, "force_int": force_int}
        obs = self._get_obs(ps, info)
        return state.replace(pipeline_state=ps, obs=obs, reward=reward,
                             metrics={"reward": reward}, info=info)

    def _target(self, step):
        # scan target sweeps xi from the reset coord; the probe slides on the
        # hfield with real friction (no controller-side cap any more).
        t = self._xi0 + jnp.clip(step.astype(jnp.float32) * self._config.scan_rate, 0.0, self._config.scan_span)
        return t, self._eta0

    # --- constraint-manifold geometry: clean-state path/force manifold (no mjx) ---
    # General env hooks (any manifold-aware solver may call them): the clean-state
    # manifold C(U)=0 this task exposes is "on the scan path AND at f_target force" —
    # surface coords (h_surf) plus the desired normal force F_n -> f_target.
    # ``manifold_residual`` is the flattened residual C(U) over a node-control
    # trajectory; ``manifold_geometry`` is its tangent geometry ∂(½‖C‖²)/∂U. Putting
    # the force on the manifold is what lets a tangent-denoising / retraction solver
    # actively reduce FORCE error vs a reward-only baseline (idea.txt RQ3). VALID
    # because the controller is closed-loop force-controlled (kp_force): the commanded
    # F_n equals the REAL contact force, so driving the command to f_target drives the
    # real force to f_target.
    def _manifold_res_node(self, u, xi0, eta0, xi_t, eta_t):
        cfg = self._config
        f_span = max(cfg.f_max - cfg.f_min, 1e-6)
        r = u[self.spec.r_slice]
        nu_raw = u[self.spec.nu_slice]
        xi = xi0 + cfg.coord_scale * r[0]
        eta = eta0 + cfg.coord_scale * r[1]
        F_n = self._force_cmd(nu_raw[0])
        return jnp.array([xi - xi_t, eta - eta_t, (F_n - cfg.f_target) / f_span])

    def manifold_residual(self, state, Ybar_nodes):
        xi0, eta0 = state.info["xi"], state.info["eta"]
        xi_t, eta_t = self._target(state.info["step"])
        return jax.vmap(lambda u: self._manifold_res_node(u, xi0, eta0, xi_t, eta_t))(Ybar_nodes).reshape(-1)

    def manifold_geometry(self, state, Ybar_nodes, t0):
        xi0, eta0 = state.info["xi"], state.info["eta"]
        xi_t, eta_t = self._target(state.info["step"])
        sq = lambda u: 0.5 * jnp.sum(self._manifold_res_node(u, xi0, eta0, xi_t, eta_t) ** 2)
        return jax.vmap(jax.grad(sq))(Ybar_nodes)

    def _reward(self, ps, xi, eta, F_n, s_vec, prev_s, step):
        cfg = self._config
        _, n_s, _, R_d = self._desired_pose(xi, eta)
        R_h = ps.site_xmat[self._ee_site].reshape(3, 3)
        xi_s, eta_s = self._target(step)
        path = (xi - xi_s) ** 2 + (eta - eta_s) ** 2
        force = (self._contact_force(ps) - cfg.f_target) ** 2   # REAL contact-force tracking
        normal = jnp.sum((R_h[:, 2] + n_s) ** 2)
        kreg = jnp.sum((s_vec - self._s_ref) ** 2)
        dk = jnp.sum((s_vec - prev_s) ** 2)
        J = cfg.w_path * path + cfg.w_F * force + cfg.w_R * normal + cfg.w_K * kreg + cfg.w_dK * dk
        return -J

    def _get_obs(self, ps, info) -> jax.Array:
        p_h = ps.site_xpos[self._ee_site]
        return jnp.concatenate([ps.qpos, ps.qvel, p_h,
                                jnp.array([info["xi"], info["eta"], info["psi"],
                                           self._contact_force(ps), self._mu])])

    # --- MDAC soft-feasibility: h_surf (eq 727) + h_normal (eq 731) + g_force ---
    def constraint_residual(self, state, action, ctx=None):
        cfg = self._config
        ps = state.pipeline_state
        xi, eta = state.info["xi"], state.info["eta"]
        _, n_s, p_d, _ = self._desired_pose(xi, eta)
        p_h = ps.site_xpos[self._ee_site]
        R_h = ps.site_xmat[self._ee_site].reshape(3, 3)
        h_surf = p_h - p_d                                  # on-surface (eq 727)
        h_normal = R_h[:, 2] + n_s                          # z aligned to -n_s (eq 731)
        h = jnp.concatenate([h_surf, h_normal])             # 6 equalities
        F_n_cmd = self._force_cmd(action[self.spec.nu_slice][0])
        g = jnp.array([F_n_cmd - cfg.f_max, cfg.f_min - F_n_cmd])   # F_min ≤ F_n ≤ F_max
        return h, g
