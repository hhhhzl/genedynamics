"""Franka-Panda surface-contact scanning env (idea.txt Exp I).

A 7-DoF Panda performs contact-rich scanning over a parametric surface. The env
is a brax ``PipelineEnv`` (mjx) over the vendored mesh-free Panda
(``assets/franka_panda/panda_arm.xml``), consuming the MGA lower-control
position-stiffness primitive

    u^arm = (Δξ, Δη, Δψ, S_h, F_n^d),   K_h = exp(S_h) ∈ S³₊₊  (3×3 translational)

``action_size = 10`` via ``core/control.PrimitiveSpec(3, 3, 1)`` (3 surface-coord
increments + svec(3×3)=6 + 1 force). The simulated contact probe is spherical,
so its default orientation stiffness is zero; ``k_orient`` remains available
only for explicit non-spherical-tool ablations. The 6×6 full task-space
stiffness (action 25) is an appendix scalability ablation, not the main primitive.

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
from genedynamics.core.control.cartesian_impedance import (
    end_effector_kinematics,
    map_cartesian_wrench,
    orientation_error,
)
from genedynamics.envs.composition import MjcfSceneComposer, SphericalToolSpec
from genedynamics.core.coverage import surface_geometry as sg
from genedynamics.core.contact.elastic_foundation import stiffness_field
from genedynamics.robots import RobotBinding, get_robot_registry
from genedynamics.robots.profile import (
    CARTESIAN_JACOBIAN, FIXED_BASE, SINGLE_TOOL, TORQUE_CONTROL,
)

_STIFF_D = 3                             # 3×3 translational stiffness (main primitive)
_SCAN_TOOL = SphericalToolSpec(tool_id="scan_probe", radius=0.02)


def _ee_home_fk(profile):
    """End-effector world position at the home pose (raw mujoco FK; needed to
    place the surface before the contact model is built)."""
    mj0 = mujoco.MjModel.from_xml_path(profile.model_path())
    d0 = mujoco.MjData(mj0)
    home = np.asarray(profile.controller_defaults["home_qpos"], dtype=np.float64)
    d0.qpos[:home.size] = home
    mujoco.mj_forward(mj0, d0)
    site = profile.elements["tool_mount"].name
    sid = mujoco.mj_name2id(mj0, mujoco.mjtObj.mjOBJ_SITE.value, site)
    return np.asarray(d0.site_xpos[sid])


def _build_contact_model(
    profile, surface, ee0, n, depth, friction, solref, collidable=True,
):
    """Place the scan start ``depth`` from the home EE along its local normal.

    The controlled EE site is the *centre* of a spherical probe, whose desired
    position is ``surface_point + probe_radius * normal``.  Aligning only the
    surface point's x/y with the EE creates a large artificial tangential reset
    error on tilted cylinder/NURBS patches.  Normal placement preserves the
    intended contact gap while making the reset error purely normal.

    The translated surface is triangulated into a MuJoCo HFIELD and injected
    with the collidable probe.  Hybrid contact keeps the probe non-collidable
    because its spatial Winkler reaction is applied by the impedance layer.
    """
    p0 = np.asarray(sg.point(surface, 0.1, 0.5))
    n0 = np.asarray(sg.normal(surface, 0.1, 0.5), dtype=np.float64)
    n0 = n0 * np.sign(n0[2] + 1e-9)
    n0 = n0 / max(float(np.linalg.norm(n0)), 1e-9)
    scan_start = np.asarray(ee0, dtype=np.float64) - float(depth) * n0
    surface = sg.translate(surface, scan_start - p0)
    g = np.linspace(0.0, 1.0, n)
    P = np.array([[np.asarray(sg.point(surface, xi, eta)) for eta in g] for xi in g])  # P[i=xi, j=eta]
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    zlo, zhi = float(Z.min()), float(Z.max()); elev = max(zhi - zlo, 1e-3)
    cx, cy = (float(X.min()) + float(X.max())) / 2, (float(Y.min()) + float(Y.max())) / 2
    rx, ry = (float(X.max()) - float(X.min())) / 2, (float(Y.max()) - float(Y.min())) / 2
    hdata = ((Z - zlo) / elev).T.reshape(-1).astype(np.float32)        # mujoco hfield: [row=y, col=x]

    composer = MjcfSceneComposer(profile)
    if profile.controller_defaults.get("strip_mesh_geoms", False):
        composer.strip_mesh_geoms()
    if profile.controller_defaults.get("replace_actuators_with_motors", False):
        composer.replace_actuators_with_motors(
            profile.controller_defaults["torque_limits"]
        )
    composer.add_spherical_tool(
        _SCAN_TOOL, friction=friction, collidable=collidable
    )
    composer.add_hfield_surface(
        name="surf",
        nrow=n,
        ncol=n,
        size=(rx, ry, elev, 0.05),
        position=(cx, cy, zlo),
        friction=friction,
        solref=solref,
    )
    mj = composer.compile()
    mj.hfield_data[:] = hdata
    return mj, surface


@dataclass
class SurfaceScanConfig:
    robot: str = "panda"
    tool: str = "scan_probe"
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
    # Optional task-space gravity compensation projected into the surface tangent
    # plane.  This removes the lateral static load that otherwise displaces a
    # 200 N/m impedance by several millimetres, while deliberately preserving
    # the normal component used for passive contact maintenance.  Default zero
    # preserves all legacy experiments; staged scan configs opt in explicitly.
    tangent_grav_comp: float = 0.0
    # Cross-path-only bias cancellation. Positive gain subtracts the estimated
    # task-space bias along the desired eta tangent, leaving scan and normal
    # directions unchanged.
    # Optional surface-tangent integral action for rejecting steady execution
    # bias without changing the normal contact force. The state is a bounded
    # Cartesian force (N); default zero preserves legacy impedance behavior.
    ki_tangent: float = 0.0
    tangent_int_max: float = 3.0
    # The contact geom is a sphere, so orientation torque is physically
    # unnecessary and couples arm rotation into the translational scan response.
    # Non-spherical tool ablations must opt in explicitly.
    k_orient: float = 0.0
    s_ref_diag: float = 5.3         # log-stiffness reference (K_ref ≈ exp(5.3) ≈ 200 N/m)
    s_scale: float = 2.0            # normalized svec [-1,1] -> log-stiffness offset (exp(3.3..7.3))
    # stiffness chart: "log_spd" K=exp(S) | "euclid" diag softplus | "fixed" K_ref | "none" I
    stiffness_mode: str = "log_spd"
    # contact MEDIUM: "rigid" (mjx hfield, default — byte-identical to before) | "soft"
    # (compliant contact = a softer solref derived from soft_stiffness, so the probe sinks
    # into the surface under the press = real deformation) | "hybrid" (spatial stiffness
    # map — added with the geometry/impedance layer later). Medium-gated, so rigid is unchanged.
    medium: str = "rigid"
    soft_stiffness: float = 1.5e3       # soft-medium contact stiffness -> solref time const
                                        # (lower = softer = more penetration / deformation)
    # hybrid medium: a fixed SPATIAL stiffness map (mjx per-geom solref can't do this, so
    # hybrid uses a model-based Winkler foundation — probe non-collidable + analytic reaction
    # k_map(ξ,η)·penetration). stiffness_map: uniform | stripes | center_hard | center_soft.
    stiffness_map: str = "uniform"
    k_hard: float = 8.0e3               # hard-region Winkler stiffness (N/m) -> small sink
    k_soft: float = 2.0e3               # soft-region Winkler stiffness -> large sink
    # Optional affine chart from task scan coordinates into the normalized
    # material map.  Defaults preserve the historical [0,1] map.  The paper
    # hybrid suites map their finite 100-step scan segment onto [0,1], so every
    # run actually crosses the declared stiffness transitions without raising
    # scan_rate beyond the impedance execution bandwidth.
    stiffness_map_xi_origin: float = 0.0
    stiffness_map_xi_span: float = 1.0
    stiffness_transition_width: float = 0.0
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
    # Optional dimensionless reward scales. Non-positive values preserve the
    # legacy raw-coordinate / raw-Newton costs exactly.
    reward_path_scale: float = 0.0
    reward_force_scale: float = 0.0
    # Tangential error of the REAL end-effector against the time-indexed scan
    # reference.  This is deliberately separate from ``w_path``: ``w_path``
    # scores command coordinates, whereas this term makes a command that races
    # ahead of the physical probe expensive.  Normal indentation is projected
    # out so compliant contact is not mistaken for path-tracking failure.
    # Default zero preserves every legacy experiment.
    w_realized_path: float = 0.0
    # Dense online counterpart of maximum realized path progress.  Reward is
    # earned only when an in-contact, on-path EE sample advances the historical
    # maximum surface coordinate; staying, retreating, and revisiting cannot
    # farm it.  Default zero preserves all pre-RL objectives.
    w_realized_progress: float = 0.0
    # Optional realized-risk terms used by the diagnostic objective. Defaults
    # preserve the original reward byte-for-byte.
    w_deformation: float = 0.0
    w_contact_loss: float = 0.0
    w_force_violation: float = 0.0
    deformation_safe: float = 0.005
    deformation_scale: float = 0.005
    metric_acquisition_steps: int = 5
    metric_path_tolerance: float = 0.005
    # Observation contract. ``legacy`` preserves the original 22-D vector.
    # ``rl_realized`` exposes only measurable task/realization signals needed by
    # the shared RL prior; it deliberately excludes the simulator's exact
    # friction, stiffness, and medium label.
    observation_mode: str = "legacy"
    # CLEAN manifold composition. MGA includes commanded force; the
    # position-only diagnostic keeps only reliable surface coordinates.
    clean_manifold_force: bool = True
    # Realization-reliability gate. These are task parameters because their
    # signals (EE tracking, contact force, deformation) belong to this env; the
    # MGA backend only consumes the resulting action-space gate.
    geometry_gate_floor: float = 0.02
    geometry_gate_path_scale: float = 0.02
    geometry_gate_normal_scale: float = 0.005
    geometry_gate_eta_path: float = 1.0
    geometry_gate_eta_normal: float = 1.0
    geometry_gate_eta_force: float = 2.0
    geometry_gate_eta_deformation: float = 1.0
    geometry_gate_eta_contact: float = 2.0
    # Frozen local realization compensation used by the staged MGA geometry.
    # The current real EE tracking bias is expressed in surface coordinates via
    # the analytic surface Jacobian and held fixed over one MPC horizon. The
    # hook is opt-in at the method layer; these values do not affect legacy runs.
    realization_compensation_gain: float = 1.0
    realization_compensation_max_coord: float = 0.1
    realization_compensation_reg: float = 1.0e-5
    realization_compensation_along_weight: float = 1.0
    realization_compensation_cross_weight: float = 1.0
    # Short-horizon true-dynamics probe used by the controllability-aware staged
    # method. Only the two surface-coordinate action channels are perturbed.
    realization_probe_horizon: int = 5
    realization_probe_eps: float = 0.02
    realization_control_gain: float = 1.0
    realization_control_max_action: float = 0.25
    realization_control_reg: float = 1.0e-4
    realization_control_along_weight: float = 1.0
    realization_control_cross_weight: float = 1.0
    # AL h_surf measures the TANGENTIAL deviation only (project out the normal penetration the
    # compliant contact naturally has) so soft-feasibility stops fighting the sink. Realized path
    # (reads pipeline_state) -> not clean-state-limited. Gated (default off = original); on soft
    # it lifts the AL methods (dial/mga), validated: it stops the AL penalizing the natural sink.
    h_surf_tangential: bool = False
    # Soft-contact manifold (eq:soft_contact_residual): the realized contact residual targets the
    # DEFORMED contact point S_0 − δ*·n instead of the undeformed surface, with target indentation
    # δ* = f_target/(k̂+ε_k) and the local stiffness k̂ = F_real/(δ_real+ε_δ) ESTIMATED from the
    # realized rollout (handles unknown / spatially-varying compliance: soft DR, hybrid maps). The
    # rigid limit k̂→∞ ⇒ δ*→0 recovers the rigid residual. Replaces the rigid h_surf in the AL seam.
    soft_contact_manifold: bool = False
    eps_delta: float = 1.0e-4
    eps_k: float = 1.0
    delta_max: float = 0.03         # 0 ≤ δ* ≤ δ_max bound (eq) — caps the indentation target so the
                                    # out-of-contact estimate (k̂→0) can't blow up the residual


class SurfaceScanEnv(PipelineEnv):
    """Robot-parameterized surface-contact scanning task."""

    def __init__(self, config: SurfaceScanConfig | None = None, **kw):
        if not BRAX_AVAILABLE:
            raise ImportError("brax + mujoco(mjx) required for PandaSurfaceScanEnv.")
        cfg = config or SurfaceScanConfig(**kw)
        self._config = cfg
        self._robot_profile = get_robot_registry().get_profile(
            "manipulator", str(cfg.robot).lower()
        )
        if self._robot_profile is None:
            raise ValueError(f"unknown manipulator robot profile: {cfg.robot}")
        self._robot_profile.require({
            FIXED_BASE, CARTESIAN_JACOBIAN, TORQUE_CONTROL, SINGLE_TOOL,
        })
        if cfg.tool != _SCAN_TOOL.tool_id:
            raise ValueError(f"unsupported surface-scan tool: {cfg.tool}")
        self._tool_spec = _SCAN_TOOL
        self._home_qpos = jnp.asarray(
            self._robot_profile.controller_defaults["home_qpos"], jnp.float32
        )
        self._xi0, self._eta0 = jnp.float32(0.1), jnp.float32(0.5)

        # domain randomization (unseen): contact friction + surface compliance.
        mu, solref = cfg.friction, cfg.contact_solref
        stiffness_draw = float(cfg.surface_stiffness)
        if str(cfg.level).lower() == "unseen":
            rk, rm = jax.random.split(jax.random.PRNGKey(cfg.surface_seed + 9973))
            lo, hi = cfg.s4_friction_range
            mu = float(jax.random.uniform(rm, minval=lo, maxval=hi))
            lo, hi = cfg.s4_stiffness_range            # softer surface -> slower solref time const
            stiffness_draw = float(jax.random.uniform(rk, minval=lo, maxval=hi))
            solref = f"{0.02 * (1.0e4 / stiffness_draw):.4f} 1"

        # contact MEDIUM = soft: a softer (more compliant) contact via a solref time
        # constant derived from soft_stiffness (same ks->solref map as the unseen DR), so
        # the probe penetrates more under the same force = surface deformation. Applied to
        # ALL surface families. (hybrid's spatial stiffness map needs the geometry/impedance
        # layer, added later; rigid leaves solref untouched.)
        if str(cfg.medium).lower() == "soft":
            # Preserve the held-out compliance draw on unseen surfaces. The old
            # ordering replaced it with one fixed soft_stiffness value.
            if str(cfg.level).lower() != "unseen":
                stiffness_draw = float(cfg.soft_stiffness)
            solref = f"{0.02 * (1.0e4 / max(stiffness_draw, 1.0)):.4f} 1"

        self._contact_friction = float(mu)
        self._contact_solref = str(solref)
        self._contact_stiffness_draw = float(stiffness_draw)

        # contact medium: rigid/soft use the REAL mjx probe-hfield contact; hybrid uses a
        # model-based Winkler reaction (probe non-collidable) for its spatial stiffness map.
        self._medium = str(cfg.medium).lower()
        self._map_kind = str(cfg.stiffness_map).lower()
        self._k_hard, self._k_soft = jnp.float32(cfg.k_hard), jnp.float32(cfg.k_soft)

        # build the REAL contact model: analytic surface -> mjx hfield, EE -> probe.
        surface = sg.surface_for_level(cfg.level, cfg.surface_seed)
        ee0 = _ee_home_fk(self._robot_profile)
        mj, self.surface = _build_contact_model(
            self._robot_profile, surface, ee0, cfg.hfield_n, cfg.contact_depth,
            mu, solref, collidable=(self._medium != "hybrid"),
        )
        sys = mjcf.load_model(mj)
        n_frames = max(1, int(round(cfg.dt / cfg.timestep)))
        super().__init__(sys=sys, backend="mjx", n_frames=n_frames)

        self._mjx_model = mjx.put_model(mj)            # for support.contact_force
        self.spec = PrimitiveSpec(pos_dim=3, stiff_dim=_STIFF_D, feed_dim=1)   # -> 10
        self._robot_binding = RobotBinding.from_mujoco_model(
            self._robot_profile, mj
        )
        self._ee_site = self._robot_binding.element_ids["tool_mount"]
        self._ee_body = int(self.sys.site_bodyid[self._ee_site])
        self._probe_geom = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM.value, "probe")
        self._surf_geom = mujoco.mj_name2id(
            mj, mujoco.mjtObj.mjOBJ_GEOM.value, "surf"
        )
        diag_idx = jnp.cumsum(jnp.arange(_STIFF_D, 0, -1)) - jnp.arange(_STIFF_D, 0, -1)
        self._s_ref = jnp.zeros((self.spec.stiff_width,), jnp.float32).at[diag_idx].set(cfg.s_ref_diag)
        self._mu = jnp.float32(mu)
        # representative contact stiffness (obs/back-compat): soft -> soft_stiffness;
        # hybrid -> mean(k_hard,k_soft); rigid -> surface_stiffness.
        self._k_surf = jnp.float32(
            cfg.soft_stiffness if self._medium == "soft"
            else 0.5 * (cfg.k_hard + cfg.k_soft) if self._medium == "hybrid"
            else cfg.surface_stiffness)

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

    # --- surface deformation: probe penetration depth into the (compliant) surface ---
    def _penetration(self, ps) -> jnp.ndarray:
        """Deepest probe penetration into the surface (mjx contact ``dist`` < 0 = the
        surface yielding under the press). 0 on rigid contact, > 0 on a soft medium —
        the per-step deformation signal the soft metrics consume."""
        c = ps.contact
        if c is None or c.dist.shape[0] == 0:
            return jnp.float32(0.0)
        on_probe = (c.geom[:, 0] == self._probe_geom) | (c.geom[:, 1] == self._probe_geom)
        pen = jnp.where(on_probe & (c.dist < 0), -c.dist, 0.0)
        return jnp.max(pen)

    # --- hybrid medium: model-based Winkler foundation (spatial stiffness map) ---
    # mjx solref is per-geom only, so a SPATIAL stiffness map is realized analytically:
    # the probe is non-collidable and the surface reacts with k(ξ,η)·penetration (applied
    # in the impedance torque). Dispatch keeps rigid/soft on the real mjx contact.
    def _k_surf_fn(self, xi, eta) -> jnp.ndarray:
        """Local surface stiffness k(ξ,η) from the fixed stiffness map (hybrid)."""
        cfg = self._config
        map_xi = jnp.clip(
            (xi - cfg.stiffness_map_xi_origin)
            / max(cfg.stiffness_map_xi_span, 1.0e-6),
            0.0,
            1.0,
        )
        return stiffness_field(
            self._map_kind,
            map_xi,
            eta,
            self._k_hard,
            self._k_soft,
            cfg.stiffness_transition_width,
        )

    def _winkler_at(self, ps, xi, eta):
        """Probe penetration δ below the rest surface and the Winkler reaction
        F = k(ξ,η)·max(0,δ) at the scan coords (hybrid, no mjx contact)."""
        _, n_s, p_d, _ = self._desired_pose(xi, eta)
        p_h = ps.site_xpos[self._ee_site]
        delta = jnp.maximum(jnp.dot(p_d - p_h, n_s), 0.0)
        return self._k_surf_fn(xi, eta) * delta, delta

    def _contact_force_at(self, ps, xi, eta) -> jnp.ndarray:
        """Normal contact force: real mjx contact (rigid/soft) or the Winkler reaction (hybrid)."""
        if self._medium == "hybrid":
            return self._winkler_at(ps, xi, eta)[0]
        return self._contact_force(ps)

    def _penetration_at(self, ps, xi, eta) -> jnp.ndarray:
        """Surface deformation: mjx penetration (rigid/soft) or the Winkler δ (hybrid)."""
        if self._medium == "hybrid":
            return self._winkler_at(ps, xi, eta)[1]
        return self._penetration(ps)

    # --- action is the MGA primitive, not the 7 joint torques ---
    @property
    def action_size(self) -> int:
        return self.spec.total_width                       # 10

    @property
    def reliability_feature_size(self) -> int:
        return 12

    @property
    def manifold_constraint_size(self) -> int:
        """Per-action equality dimension of ``manifold_residual``."""
        return 3 if self._config.clean_manifold_force else 2

    @property
    def inequality_constraint_size(self) -> int:
        """Per-action inequality dimension of ``constraint_residual``."""
        return 2

    def reset(self, rng: jax.Array) -> State:
        ps = self.pipeline_init(self._home_qpos, jnp.zeros(self.sys.qd_size()))
        info = {
            "xi": self._xi0, "eta": self._eta0, "psi": jnp.float32(0.0),
            "prev_s": self._s_ref, "step": jnp.int32(0),
            "max_realized_xi": self._xi0,
            "force_int": jnp.float32(0.0),     # contact-force integral (admittance state)
            "tangent_force_int": jnp.zeros((3,), dtype=jnp.float32),
        }
        if self._config.observation_mode == "rl_realized":
            info["prev_action"] = jnp.zeros((self.action_size,), dtype=jnp.float32)
            realized_coords, _ = self._realized_surface_coords(
                ps, info["xi"], info["eta"]
            )
            info["realized_coord_error"] = realized_coords - jnp.asarray([
                info["xi"], info["eta"]
            ])
        obs = self._get_obs(ps, info)
        return State(ps, obs, jnp.float32(0.0), jnp.float32(0.0),
                     {
                         "reward": jnp.float32(0.0),
                         "realized_progress": jnp.float32(0.0),
                     }, info)

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
        p_d = p_s + (self._config.standoff + self._tool_spec.radius) * n_s
        z_d = -n_s
        t_xi, _ = sg.tangents(self.surface, xi, eta)
        x_d = t_xi / (jnp.linalg.norm(t_xi) + 1e-9)
        y_d = jnp.cross(z_d, x_d); y_d = y_d / (jnp.linalg.norm(y_d) + 1e-9)
        x_d = jnp.cross(y_d, z_d)
        R_d = jnp.stack([x_d, y_d, z_d], axis=1)
        return p_s, n_s, p_d, R_d

    def _ee_kin(self, ps):
        return end_effector_kinematics(
            self.sys, ps, self._ee_site, self._ee_body
        )

    @staticmethod
    def _orient_err(R_h, R_d):
        return orientation_error(R_h, R_d)

    # --- 3×3 translational stiffness chart (flag-driven; stiffness ablations) ---
    def _stiffness(self, s_vec):
        mode = self._config.stiffness_mode
        if mode == "fixed":
            return stiffness_log_to_pd(self._s_ref, _STIFF_D)   # K_ref, ignore S block
        if mode == "none":
            return jnp.eye(_STIFF_D, dtype=jnp.float32)
        if mode == "euclid":                                    # Euclidean SPD (not log)
            from genedynamics.core.control.stiffness import svec2sym
            diag = jnp.diagonal(svec2sym(s_vec, _STIFF_D))
            scale = max(self._config.s_scale, 1.0e-6)
            raw = jnp.clip(
                (diag - self._config.s_ref_diag) / scale, -1.0, 1.0
            )
            k_ref = jnp.exp(self._config.s_ref_diag)
            k_lo = jnp.exp(self._config.s_ref_diag - self._config.s_scale)
            k_hi = jnp.exp(self._config.s_ref_diag + self._config.s_scale)
            # Direct Euclidean/clamped diagonal chart with the same physical
            # reference and endpoint range as Log-SPD.  Piecewise affine
            # interpolation keeps raw=0 exactly at K_ref; the previous
            # softplus(S≈6.4) implementation silently changed 602 N/m into
            # 6.4 N/m and was not a fair parameterization ablation.
            d = jnp.where(
                raw >= 0.0,
                k_ref + raw * (k_hi - k_ref),
                k_ref + raw * (k_ref - k_lo),
            )
            return jnp.diag(d)
        return stiffness_log_to_pd(s_vec, _STIFF_D)             # log_spd: K = exp(S)

    # --- impedance π_low: primitive -> joint torque (eq 698–724) ---
    # Translational impedance K_t(S) on position + FIXED k_orient on orientation +
    # an F_n feedforward pressing into the surface. Coulomb friction is now REAL
    # (the mjx probe-hfield contact), so no controller-side tangential cap.
    def _impedance_tau(
        self, ps, n_s, p_d, R_d, s_vec, F_n, force_int,
        tangent_force_int, k_local=None,
    ):
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
        # contact-force readback: real mjx contact (rigid/soft) OR the model-based Winkler
        # reaction k(ξ,η)·δ (hybrid — probe non-collidable, so the reaction is added below).
        if k_local is not None:
            delta = jnp.maximum(jnp.dot(p_d - p_h, n_s), 0.0)
            F_meas = k_local * delta
        else:
            F_meas = self._contact_force(ps)
        dofs = jnp.asarray(self._robot_binding.dof_indices)
        g_ee = jnp.linalg.solve(
            jacp.T @ jacp + 1e-6 * jnp.eye(3),
            jacp.T @ ps.qfrc_bias[dofs],
        )
        g_n = jnp.dot(g_ee, n_s)                              # arm-weight normal force at EE
        g_tangent = g_ee - g_n * n_s
        t_cross = R_d[:, 1]
        g_cross = jnp.dot(g_ee, t_cross) * t_cross
        # Only the UNCOMPENSATED fraction of the normal gravity load remains
        # available as passive press. Subtracting the full g_n when
        # grav_comp>0 double-cancels gravity and makes the probe float off.
        residual_g_n = (1.0 - cfg.grav_comp) * g_n
        F_eff = jnp.clip(
            F_n - residual_g_n
            + cfg.kp_force * (F_n - F_meas)
            + force_int,
                         cfg.f_min - cfg.f_cmd_pad, cfg.f_max + cfg.f_cmd_pad)
        f_pos = (
            K_t @ (p_d - p_h)
            + tangent_force_int
            - cfg.d_damp * lin_v
            - F_eff * n_s
        )  # press INTO surface (-n_s)
        # Cancel only the surface-tangent component of the task-space gravity
        # load.  The normal component remains in the real dynamics and continues
        # to provide the contact-maintenance bias described above.
        f_pos = f_pos + cfg.tangent_grav_comp * g_tangent
        if k_local is not None:
            f_pos = f_pos + F_meas * n_s                      # Winkler surface reaction (out, +n_s)
        m_rot = cfg.k_orient * self._orient_err(R_h, R_d) - cfg.d_damp * ang_v
        # PARTIAL gravity/coriolis compensation: full comp makes the arm weightless
        # and it floats off; zero comp makes it lean hard (~20 N). A small residual
        # weight keeps light contact and the planner trims F_n to hit f_target.
        return map_cartesian_wrench(
            jacp,
            jacr,
            f_pos,
            m_rot,
            ps.qfrc_bias[dofs],
            gravity_compensation=cfg.grav_comp,
            torque_limits=self._robot_profile.controller_defaults["torque_limits"],
        )

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
        # hybrid: the local surface stiffness for the model-based Winkler reaction (held for
        # the control period). None for rigid/soft (real mjx contact, no manual reaction).
        k_local = self._k_surf_fn(xi, eta) if self._medium == "hybrid" else None

        # FAST INNER FORCE/IMPEDANCE SERVO: recompute the controller (and the force
        # integral) EVERY physics substep instead of holding one torque open-loop for the
        # whole control period. This is the real-robot hierarchy -- a fast (here per-
        # substep, ~200 Hz) force loop under the ~50 Hz MPC -- which regulates the contact
        # force WITHIN the control step (rejecting the following/contact disturbances the
        # MPC-rate loop is too slow for). The MPC command (xi, eta, S_vec, F_n) is held.
        if cfg.fast_force_loop:
            def _substep(carry, _):
                ps_i, fint, tint = carry
                tau_i = self._impedance_tau(
                    ps_i, n_s, p_d, R_d, S_vec, F_n, fint, tint, k_local
                )
                ps_i = self._pipeline.step(self.sys, ps_i, tau_i, self._debug)
                fint = jnp.clip(fint + cfg.ki_force * (F_n - self._contact_force_at(ps_i, xi, eta)),
                                -cfg.force_int_max, cfg.force_int_max)
                p_i = ps_i.site_xpos[self._ee_site]
                e_pos = p_d - p_i
                e_tangent = e_pos - jnp.dot(e_pos, n_s) * n_s
                tint = jnp.clip(
                    tint + cfg.ki_tangent * cfg.timestep * e_tangent,
                    -cfg.tangent_int_max,
                    cfg.tangent_int_max,
                )
                return (ps_i, fint, tint), None
            (ps, force_int, tangent_force_int), _ = jax.lax.scan(
                _substep,
                (
                    state.pipeline_state,
                    state.info["force_int"],
                    state.info["tangent_force_int"],
                ),
                (),
                self._n_frames,
            )
        else:
            force_int = state.info["force_int"]
            tangent_force_int = state.info["tangent_force_int"]
            tau = self._impedance_tau(
                state.pipeline_state,
                n_s,
                p_d,
                R_d,
                S_vec,
                F_n,
                force_int,
                tangent_force_int,
                k_local,
            )
            ps = self.pipeline_step(state.pipeline_state, tau)
            force_int = jnp.clip(force_int + cfg.ki_force * (F_n - self._contact_force_at(ps, xi, eta)),
                                 -cfg.force_int_max, cfg.force_int_max)
            p_i = ps.site_xpos[self._ee_site]
            e_pos = p_d - p_i
            e_tangent = e_pos - jnp.dot(e_pos, n_s) * n_s
            tangent_force_int = jnp.clip(
                tangent_force_int + cfg.ki_tangent * cfg.dt * e_tangent,
                -cfg.tangent_int_max,
                cfg.tangent_int_max,
            )

        reward = self._reward(ps, xi, eta, F_n, S_vec, state.info["prev_s"], state.info["step"])
        max_realized_xi = state.info["max_realized_xi"]
        realized_progress = jnp.float32(0.0)
        realized_coords = None
        tangent_error = None
        if cfg.w_realized_progress or cfg.observation_mode == "rl_realized":
            realized_coords, tangent_error = self._realized_surface_coords(
                ps, xi, eta
            )
        if cfg.w_realized_progress:
            in_contact = self._contact_force_at(ps, xi, eta) > 0.5
            on_path = jnp.linalg.norm(tangent_error) <= cfg.track_tol
            current_target_xi, _ = self._target(state.info["step"])
            candidate = jnp.where(
                in_contact & on_path,
                jnp.clip(realized_coords[0], self._xi0, current_target_xi),
                max_realized_xi,
            )
            next_max = jnp.maximum(max_realized_xi, candidate)
            realized_progress = (
                next_max - max_realized_xi
            ) / max(cfg.scan_rate, 1e-6)
            reward = reward + cfg.w_realized_progress * realized_progress
            max_realized_xi = next_max
        # MERGE (not replace) so brax training wrappers' info keys (steps/truncation/
        # episode_metrics/...) survive the step; a no-op for the unwrapped MGA/baseline path.
        info = {**state.info, "xi": xi, "eta": eta, "psi": psi, "prev_s": S_vec,
                "step": state.info["step"] + 1, "force_int": force_int,
                "tangent_force_int": tangent_force_int,
                "max_realized_xi": max_realized_xi}
        if cfg.observation_mode == "rl_realized":
            info["prev_action"] = jnp.clip(action, -1.0, 1.0)
            info["realized_coord_error"] = (
                realized_coords - jnp.asarray([xi, eta])
            )
        obs = self._get_obs(ps, info)
        return state.replace(
            pipeline_state=ps,
            obs=obs,
            reward=reward,
            metrics={
                "reward": reward,
                "realized_progress": realized_progress,
            },
            info=info,
        )

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
        xi = xi0 + cfg.coord_scale * r[0]
        eta = eta0 + cfg.coord_scale * r[1]
        pos = jnp.array([xi - xi_t, eta - eta_t])
        if not cfg.clean_manifold_force:
            return pos
        nu_raw = u[self.spec.nu_slice]
        F_n = self._force_cmd(nu_raw[0])
        return jnp.concatenate([
            pos, jnp.array([(F_n - cfg.f_target) / f_span])
        ])

    def manifold_residual(self, state, Ybar_nodes):
        xi0, eta0 = state.info["xi"], state.info["eta"]
        xi_t, eta_t = self._target(state.info["step"])
        return jax.vmap(lambda u: self._manifold_res_node(u, xi0, eta0, xi_t, eta_t))(Ybar_nodes).reshape(-1)

    def manifold_geometry(self, state, Ybar_nodes, t0):
        xi0, eta0 = state.info["xi"], state.info["eta"]
        xi_t, eta_t = self._target(state.info["step"])
        sq = lambda u: 0.5 * jnp.sum(self._manifold_res_node(u, xi0, eta0, xi_t, eta_t) ** 2)
        return jax.vmap(jax.grad(sq))(Ybar_nodes)

    def _surface_coordinate_jacobian(self, xi, eta):
        """Analytic desired-EE position Jacobian wrt surface coordinates."""
        coords = jnp.asarray([xi, eta], jnp.float32)
        return jax.jacfwd(
            lambda z: self._desired_pose(z[0], z[1])[2]
        )(coords)

    def _realization_coordinate_offset_raw(self, state):
        """Unscaled local real-EE bias in ``(xi, eta)`` coordinates."""
        cfg = self._config
        xi, eta = state.info["xi"], state.info["eta"]
        _, n_s, p_d, _ = self._desired_pose(xi, eta)
        ee = state.pipeline_state.site_xpos[self._ee_site]
        error = ee - p_d
        tangent_error = error - jnp.dot(error, n_s) * n_s
        Jq = self._surface_coordinate_jacobian(xi, eta)
        reg = max(cfg.realization_compensation_reg, 1e-9)
        dq = jnp.linalg.solve(
            Jq.T @ Jq + reg * jnp.eye(2, dtype=Jq.dtype),
            Jq.T @ tangent_error,
        )
        in_contact = self._contact_force_at(
            state.pipeline_state, xi, eta
        ) > 0.5
        return jnp.where(in_contact, dq, jnp.zeros_like(dq))

    def realization_coordinate_offset(self, state):
        """Frozen local real-EE bias in ``(xi, eta)`` coordinates.

        If ``p_real ~= p_des(q_cmd) + e`` and
        ``p_des(q_cmd + dq) ~= p_des(q_cmd) + J_q dq``, then
        ``dq = J_q^+ e_tangent`` estimates the realized coordinate bias. A
        command targeting ``q_target - dq`` compensates it. The estimate reads
        the real state but is stop-gradient: geometry differentiation remains
        on the clean analytic command model.
        """
        cfg = self._config
        dq = self._realization_coordinate_offset_raw(state)
        dq = dq * jnp.asarray(
            [
                cfg.realization_compensation_along_weight,
                cfg.realization_compensation_cross_weight,
            ],
            dtype=dq.dtype,
        )
        limit = max(cfg.realization_compensation_max_coord, 0.0)
        dq = jnp.clip(
            cfg.realization_compensation_gain * dq, -limit, limit
        )
        return jax.lax.stop_gradient(dq)

    def _probe_final_state(self, state, actions):
        def body(s, u):
            s2 = self.step(s, u)
            return s2, None

        return jax.lax.scan(body, state, actions)[0]

    def realization_control_jacobian(self, state, dense_actions):
        """True short-horizon ``d(realized xi,eta)/d(action xi,eta)``.

        Central differences are evaluated with the existing MJX step/scan stack
        around the incumbent dense plan. The result is stop-gradient, so the
        subsequent geometry Jacobian differentiates only the clean node spline.
        """
        cfg = self._config
        horizon = min(
            max(int(cfg.realization_probe_horizon), 1),
            int(dense_actions.shape[0]),
        )
        actions = dense_actions[:horizon]
        eps = max(cfg.realization_probe_eps, 1e-5)
        basis = jnp.eye(2, dtype=dense_actions.dtype) * eps

        def final_ee(delta):
            perturbed = actions.at[
                :, self.spec.r_slice.start:self.spec.r_slice.start + 2
            ].add(delta[None, :])
            final = self._probe_final_state(state, perturbed)
            return final.pipeline_state.site_xpos[self._ee_site]

        ee_plus = jax.vmap(final_ee)(basis)
        ee_minus = jax.vmap(final_ee)(-basis)
        d_ee = (ee_plus - ee_minus) / (2.0 * eps)  # (2 action axes, 3 xyz)
        nominal_final = self._probe_final_state(state, actions)
        Jq = self._surface_coordinate_jacobian(
            nominal_final.info["xi"], nominal_final.info["eta"]
        )
        reg = max(cfg.realization_compensation_reg, 1e-9)
        pinv = jnp.linalg.solve(
            Jq.T @ Jq + reg * jnp.eye(2, dtype=Jq.dtype),
            Jq.T,
        )
        B = pinv @ d_ee.T
        return jax.lax.stop_gradient(B)

    def prepare_realization_context(
        self, state, dense_actions, *, gate_controllability=False
    ):
        """Attach a frozen short-horizon response map to the solver state."""
        B = self.realization_control_jacobian(state, dense_actions)
        error = jax.lax.stop_gradient(
            self._realization_coordinate_offset_raw(state)
        )
        realization_gate = (
            self.geometry_reliability(state)["action"][:2]
            if gate_controllability
            else jnp.ones((2,), dtype=B.dtype)
        )
        info = {
            **state.info,
            "_mga_realization_B": B,
            "_mga_realization_error": error,
            "_mga_realization_gate": jax.lax.stop_gradient(realization_gate),
        }
        return state.replace(info=info)

    def _manifold_residual_horizon_impl(
        self, state, dense_actions, t0, target_shift
    ):
        """Cumulative time-indexed residual with a frozen target offset."""
        cfg = self._config
        r = dense_actions[:, self.spec.r_slice]
        xi = jnp.clip(
            state.info["xi"] + cfg.coord_scale * jnp.cumsum(r[:, 0]), 0.0, 1.0
        )
        eta = jnp.clip(
            state.info["eta"] + cfg.coord_scale * jnp.cumsum(r[:, 1]), 0.0, 1.0
        )
        steps = jnp.asarray(t0, jnp.float32) + jnp.arange(
            dense_actions.shape[0], dtype=jnp.float32
        )
        xi_t, eta_t = jax.vmap(self._target)(steps)
        xi_t = xi_t + target_shift[:, 0]
        eta_t = eta_t + target_shift[:, 1]
        pos = jnp.stack([xi - xi_t, eta - eta_t], axis=-1).reshape(-1)
        if not cfg.clean_manifold_force:
            return pos
        f_span = max(cfg.f_max - cfg.f_min, 1e-6)
        nu_raw = dense_actions[:, self.spec.nu_slice][:, 0]
        force = jax.vmap(self._force_cmd)(nu_raw)
        return jnp.concatenate([pos, (force - cfg.f_target) / f_span])

    def manifold_residual_horizon(self, state, dense_actions, t0):
        """Cumulative, time-indexed clean-command residual over dense controls.

        Actions encode coordinate increments, so node-wise residuals cannot be
        evaluated independently against one current target. This hook remains
        task-owned; the MGA experiment adapter supplies the existing
        ``NodeSpline`` map from solver nodes to ``dense_actions``.
        """
        return self._manifold_residual_horizon_impl(
            state,
            dense_actions,
            t0,
            jnp.zeros((dense_actions.shape[0], 2), dtype=dense_actions.dtype),
        )

    def manifold_residual_horizon_realized(self, state, dense_actions, t0):
        """Horizon manifold corrected by the frozen local realization bias."""
        return self._manifold_residual_horizon_impl(
            state,
            dense_actions,
            t0,
            -jnp.broadcast_to(
                self.realization_coordinate_offset(state),
                (dense_actions.shape[0], 2),
            ),
        )

    def manifold_residual_horizon_controllable(self, state, dense_actions, t0):
        """Horizon residual lifted through a frozen true-dynamics response map."""
        cfg = self._config
        B = state.info["_mga_realization_B"]
        error = state.info["_mga_realization_error"] * jnp.asarray(
            [
                cfg.realization_control_along_weight,
                cfg.realization_control_cross_weight,
            ],
            dtype=B.dtype,
        )
        reg = max(cfg.realization_control_reg, 1e-9)
        correction = -B.T @ jnp.linalg.solve(
            B @ B.T + reg * jnp.eye(2, dtype=B.dtype),
            error,
        )
        max_action = max(cfg.realization_control_max_action, 0.0)
        correction = jnp.clip(correction, -max_action, max_action)
        probe_horizon = min(
            max(int(cfg.realization_probe_horizon), 1),
            int(dense_actions.shape[0]),
        )
        command_shift = (
            cfg.realization_control_gain
            * cfg.coord_scale
            * float(probe_horizon)
            * correction
        )
        # Reliability gates only the empirical response-map lead.  Clean
        # horizon geometry and retraction remain active, so low confidence can
        # recover instead of freezing the entire controller.  This mirrors the
        # insertion task's controllability contract and makes the staged
        # ``mga_controllable_gate`` method causal rather than diagnostic-only.
        command_shift = command_shift * state.info["_mga_realization_gate"]
        coord_limit = max(cfg.realization_compensation_max_coord, 0.0)
        command_shift = jnp.clip(command_shift, -coord_limit, coord_limit)
        ramp = jnp.clip(
            (jnp.arange(dense_actions.shape[0], dtype=dense_actions.dtype) + 1.0)
            / float(probe_horizon),
            0.0,
            1.0,
        )
        target_shift = ramp[:, None] * command_shift[None, :]
        return self._manifold_residual_horizon_impl(
            state, dense_actions, t0, target_shift
        )

    def geometry_reliability(self, state):
        """Pre-action realization-consistency scores and action-space gate.

        The component scores are method-independent observables, so the metrics
        extractor can recompute them for every trajectory. The solver decides
        whether to apply them as a scalar or component-wise gate.
        """
        cfg = self._config
        ps = state.pipeline_state
        xi, eta = state.info["xi"], state.info["eta"]
        _, n_s, p_d, _ = self._desired_pose(xi, eta)
        ee = ps.site_xpos[self._ee_site]
        h = ee - p_d
        normal_offset = jnp.dot(h, n_s)
        tangent = h - normal_offset * n_s
        f_real = self._contact_force_at(ps, xi, eta)
        deformation = self._penetration_at(ps, xi, eta)
        contact_loss = (f_real <= 0.5).astype(jnp.float32)

        path_scale = max(cfg.geometry_gate_path_scale, 1e-6)
        normal_scale = max(cfg.geometry_gate_normal_scale, 1e-6)
        force_scale = max(cfg.f_max - cfg.f_min, 1e-6)
        deformation_scale = max(cfg.deformation_scale, 1e-6)
        e_path = jnp.linalg.norm(tangent) / path_scale
        e_normal = jnp.abs(normal_offset) / normal_scale
        e_force = jnp.abs(f_real - cfg.f_target) / force_scale
        e_deformation = (
            jnp.maximum(deformation - cfg.deformation_safe, 0.0)
            / deformation_scale
        )

        floor = jnp.clip(jnp.asarray(cfg.geometry_gate_floor, jnp.float32), 0.0, 1.0)

        def gated(raw):
            return floor + (1.0 - floor) * jnp.exp(-raw)

        g_path = gated(cfg.geometry_gate_eta_path * e_path)
        g_normal = gated(
            cfg.geometry_gate_eta_normal * e_normal
            + cfg.geometry_gate_eta_contact * contact_loss
        )
        g_force = gated(
            cfg.geometry_gate_eta_force * e_force
            + cfg.geometry_gate_eta_deformation * e_deformation
            + cfg.geometry_gate_eta_contact * contact_loss
        )
        g_stiffness = jnp.sqrt(g_normal * g_force)
        g_scalar = (g_path * g_normal * g_force) ** (1.0 / 3.0)

        action_gate = jnp.ones((self.action_size,), dtype=jnp.float32)
        action_gate = action_gate.at[self.spec.r_slice.start:self.spec.r_slice.start + 2].set(g_path)
        action_gate = action_gate.at[self.spec.r_slice.start + 2].set(g_normal)
        action_gate = action_gate.at[self.spec.s_slice].set(g_stiffness)
        action_gate = action_gate.at[self.spec.nu_slice].set(g_force)
        # The controllability-aware route gates its empirical tangential lead
        # with ``action_gate[:2]`` inside the horizon residual.  Clean path
        # geometry must remain fully active for recovery, but forcing uncertain
        # normal/impedance/force blocks through projection and retraction can
        # create a peak at soft--hard switches.  This second task-owned vector
        # therefore gates only those blocks in the generic optimizer.
        clean_action_gate = jnp.ones_like(action_gate)
        clean_action_gate = clean_action_gate.at[
            self.spec.r_slice.start + 2
        ].set(g_normal)
        clean_action_gate = clean_action_gate.at[self.spec.s_slice].set(
            g_stiffness
        )
        clean_action_gate = clean_action_gate.at[self.spec.nu_slice].set(
            g_force
        )
        return {
            "action": action_gate,
            "clean_action": clean_action_gate,
            "scalar": g_scalar,
            "path": g_path,
            "normal": g_normal,
            "stiffness": g_stiffness,
            "force": g_force,
            "path_error": e_path,
            "normal_error": e_normal,
            "force_error": e_force,
            "deformation_risk": e_deformation,
            "contact_loss": contact_loss,
        }

    def reliability_features(self, state, action):
        """Observable pre-action features for learned MGA reliability.

        Keep this contract synchronized with
        :mod:`genedynamics.learning.reliability`.  No surface label, friction,
        stiffness draw, or simulator-only material parameter is exposed.
        """
        cfg = self._config
        ps = state.pipeline_state
        xi, eta = state.info["xi"], state.info["eta"]
        _, n_s, p_d, _ = self._desired_pose(xi, eta)
        ee = ps.site_xpos[self._ee_site]
        h = ee - p_d
        normal_offset = jnp.dot(h, n_s)
        tangent = h - normal_offset * n_s
        force = self._contact_force_at(ps, xi, eta)
        deformation = self._penetration_at(ps, xi, eta)
        _, _, force_cmd = self._unpack(action)
        previous = state.info["prev_action"]
        force_scale = max(cfg.f_max - cfg.f_min, 1.0e-6)
        return jnp.asarray([
            jnp.linalg.norm(tangent) / max(cfg.geometry_gate_path_scale, 1.0e-6),
            normal_offset / max(cfg.geometry_gate_normal_scale, 1.0e-6),
            (force - cfg.f_target) / force_scale,
            deformation / max(cfg.deformation_scale, 1.0e-6),
            (force <= 0.5).astype(jnp.float32),
            jnp.clip(
                state.info["step"].astype(jnp.float32) * cfg.scan_rate
                / max(cfg.scan_span, 1.0e-6),
                0.0,
                1.0,
            ),
            action[0],
            action[1],
            action[2],
            jnp.sqrt(jnp.mean(action[self.spec.s_slice] ** 2)),
            (force_cmd - cfg.f_target) / force_scale,
            jnp.sqrt(jnp.mean((action - previous) ** 2)),
        ], dtype=jnp.float32)

    def _reward(self, ps, xi, eta, F_n, s_vec, prev_s, step):
        cfg = self._config
        _, n_s, _, R_d = self._desired_pose(xi, eta)
        R_h = ps.site_xmat[self._ee_site].reshape(3, 3)
        xi_s, eta_s = self._target(step)
        path_scale = (
            cfg.reward_path_scale if cfg.reward_path_scale > 0.0 else 1.0
        )
        path = (
            ((xi - xi_s) / path_scale) ** 2
            + ((eta - eta_s) / path_scale) ** 2
        )
        F_real = self._contact_force_at(ps, xi, eta)
        force_scale = (
            cfg.reward_force_scale if cfg.reward_force_scale > 0.0 else 1.0
        )
        force = ((F_real - cfg.f_target) / force_scale) ** 2
        normal = jnp.sum((R_h[:, 2] + n_s) ** 2)
        kreg = jnp.sum((s_vec - self._s_ref) ** 2)
        dk = jnp.sum((s_vec - prev_s) ** 2)
        J = cfg.w_path * path + cfg.w_F * force + cfg.w_R * normal + cfg.w_K * kreg + cfg.w_dK * dk
        if cfg.w_realized_path:
            _, n_target, p_target, _ = self._desired_pose(xi_s, eta_s)
            realized_error = ps.site_xpos[self._ee_site] - p_target
            realized_tangent = (
                realized_error - jnp.dot(realized_error, n_target) * n_target
            )
            path_scale = max(cfg.track_tol, 1e-6)
            J = J + cfg.w_realized_path * (
                jnp.sum(realized_tangent ** 2) / (path_scale ** 2)
            )
        if cfg.w_deformation:
            penetration = self._penetration_at(ps, xi, eta)
            dscale = max(cfg.deformation_scale, 1e-6)
            deformation_risk = (
                jnp.maximum(penetration - cfg.deformation_safe, 0.0) / dscale
            ) ** 2
            J = J + cfg.w_deformation * deformation_risk
        if cfg.w_contact_loss:
            J = J + cfg.w_contact_loss * (F_real <= 0.5).astype(jnp.float32)
        if cfg.w_force_violation:
            fspan = max(cfg.f_max - cfg.f_min, 1e-6)
            force_risk = (
                jnp.maximum(F_real - cfg.f_max, 0.0) ** 2
                + jnp.maximum(cfg.f_min - F_real, 0.0) ** 2
            ) / (fspan ** 2)
            J = J + cfg.w_force_violation * force_risk
        return -J

    def _realized_surface_coords(self, ps, xi, eta):
        """Estimate measurable EE surface coordinates around the command."""
        cfg = self._config
        _, n_s, p_d, _ = self._desired_pose(xi, eta)
        ee_error = ps.site_xpos[self._ee_site] - p_d
        tangent_error = ee_error - jnp.dot(ee_error, n_s) * n_s
        Jq = self._surface_coordinate_jacobian(xi, eta)
        reg = max(cfg.realization_compensation_reg, 1e-9)
        delta = jnp.linalg.solve(
            Jq.T @ Jq + reg * jnp.eye(2, dtype=Jq.dtype),
            Jq.T @ tangent_error,
        )
        return jnp.asarray([xi, eta]) + delta, tangent_error

    def _get_obs(self, ps, info) -> jax.Array:
        p_h = ps.site_xpos[self._ee_site]
        if self._config.observation_mode == "rl_realized":
            cfg = self._config
            xi, eta = info["xi"], info["eta"]
            xi_t, eta_t = self._target(info["step"])
            coord_scale = max(
                cfg.reward_path_scale
                if cfg.reward_path_scale > 0.0 else cfg.scan_span,
                1e-6,
            )
            target_error = jnp.asarray([xi_t - xi, eta_t - eta]) / coord_scale

            realized_coord_error = info["realized_coord_error"] / coord_scale

            force = self._contact_force_at(ps, xi, eta)
            force_scale = max(
                cfg.reward_force_scale
                if cfg.reward_force_scale > 0.0
                else cfg.f_max - cfg.f_min,
                1e-6,
            )
            force_error = (force - cfg.f_target) / force_scale
            deformation = self._penetration_at(ps, xi, eta)
            deformation_scaled = deformation / max(cfg.deformation_scale, 1e-6)
            in_contact = (force > 0.5).astype(jnp.float32)
            phase = jnp.clip(
                info["step"].astype(jnp.float32) * cfg.scan_rate
                / max(cfg.scan_span, 1e-6),
                0.0,
                1.0,
            )
            task_obs = jnp.concatenate([
                target_error,
                realized_coord_error,
                jnp.asarray([
                    info["psi"],
                    force_error,
                    deformation_scaled,
                    in_contact,
                    phase,
                ]),
                info["prev_action"],
            ])
            return jnp.concatenate([ps.qpos, ps.qvel, p_h, task_obs])
        if self._config.observation_mode != "legacy":
            raise ValueError(
                "observation_mode must be 'legacy' or 'rl_realized', got "
                f"{self._config.observation_mode!r}"
            )
        return jnp.concatenate([ps.qpos, ps.qvel, p_h,
                                jnp.array([info["xi"], info["eta"], info["psi"],
                                           self._contact_force_at(ps, info["xi"], info["eta"]), self._mu])])

    def sequence_score_risk(
        self, state, actions, aug_lambda=0.0, aug_rho=0.0
    ):
        """Joint predicted score and risk for one acceptance candidate.

        Computing both in one dynamics scan avoids compiling and retaining a
        separate batch-2 reward rollout in addition to the task risk rollout.
        The score exactly matches the augmented planner reward when nonzero AL
        parameters are supplied, and the risk vector keeps the existing task-
        owned normalization.

        Returns ``(mean_score, risk)`` where risk is dimensionless
        ``[force_violation_rate, contact_loss_rate, deformation_CVaR20,
        force_MAE]``.
        """
        cfg = self._config

        def body(s, u):
            s2 = self.step(s, u)
            h, g = self.constraint_residual(s2, u)
            residual = jnp.concatenate([jnp.abs(h), jax.nn.relu(g)], axis=-1)
            penalty = (
                aug_lambda * jnp.sum(residual)
                + 0.5 * aug_rho * jnp.sum(residual * residual)
            )
            xi, eta = s2.info["xi"], s2.info["eta"]
            force = self._contact_force_at(s2.pipeline_state, xi, eta)
            deformation = self._penetration_at(s2.pipeline_state, xi, eta)
            force_violation = (
                (force < cfg.f_min) | (force > cfg.f_max)
            ).astype(jnp.float32)
            contact_loss = (force <= 0.5).astype(jnp.float32)
            force_error = (
                jnp.abs(force - cfg.f_target)
                / max(
                    cfg.reward_force_scale
                    if cfg.reward_force_scale > 0.0
                    else cfg.f_max - cfg.f_min,
                    1e-6,
                )
            )
            return s2, (
                s2.reward - penalty,
                jnp.asarray([
                    force_violation,
                    contact_loss,
                    deformation / max(cfg.deformation_scale, 1e-6),
                    force_error,
                ]),
            )

        _, (rewards, per_step) = jax.lax.scan(body, state, actions)
        tail_count = max(1, (int(actions.shape[0]) + 4) // 5)
        deformation_cvar = jnp.mean(jnp.sort(per_step[:, 2])[-tail_count:])
        risk = jnp.asarray([
            jnp.mean(per_step[:, 0]),
            jnp.mean(per_step[:, 1]),
            deformation_cvar,
            jnp.mean(per_step[:, 3]),
        ])
        return jnp.mean(rewards), risk

    def sequence_risk(self, state, actions) -> jax.Array:
        """Task-owned realized risk vector for one candidate sequence."""
        return self.sequence_score_risk(state, actions)[1]

    def sequence_risk_is_safe(self, risk):
        """Hard deployment safety for scanning candidate horizons.

        Contact retention, force-target error, and compliant deformation are
        reported performance/quality quantities for this task.  Exceeding the
        physical normal-force interval is the hard event that must never be
        deployed.  Keeping this interpretation task-owned prevents the generic
        MGA backend from guessing the Surface risk-vector schema.
        """
        return risk[0] <= 1.0e-8

    def sequence_risk_is_no_worse(self, candidate, incumbent, tolerance):
        """Compare the hard Surface risk without double-counting quality.

        Deformation, force-target error, and contact retention already enter
        the planner score with the task's declared progress--safety weights.
        Requiring each of those quality coordinates to improve monotonically
        rejects valid Pareto moves and can permanently freeze a safe shifted
        incumbent at a material transition.  Only the physical force-bound
        event is lexicographic; :meth:`sequence_risk_is_safe` independently
        requires the selected candidate to have zero predicted violations.
        """
        return candidate[0] <= incumbent[0] + tolerance[0]

    def emergency_sequence_score_risk(
        self, state, actions, aug_lambda=0.0, aug_rho=0.0
    ):
        """Certify the one action actually committed from an emergency plan."""
        return self.sequence_score_risk(
            state, actions[:1], aug_lambda, aug_rho
        )

    def emergency_plan(self, state, fallback, t0):
        """Zero-force hold used when both performance candidates are unsafe.

        Surface actions contain no normal-position retract channel: normal
        unloading is owned by the inner force loop.  The first node commands
        ``f_min`` while holding surface coordinates, so that loop unloads
        without a tangential jump at a material switch.  Tail nodes command a
        zero commanded force and the nominal slow scan increment; after
        receding shift this becomes a safe moving recovery incumbent instead
        of a sticky all-zero emergency.  Passive contact is retained by the
        uncompensated normal gravity load already used by this task, while a
        maximal diagonal SPD stiffness retracts excess deformation before a
        soft-to-hard transition.
        """
        del state, t0
        cfg = self._config
        lo = cfg.f_min - cfg.f_cmd_pad
        hi = cfg.f_max + cfg.f_cmd_pad
        raw_f_min = 2.0 * (cfg.f_min - lo) / max(hi - lo, 1.0e-6) - 1.0
        emergency = jnp.zeros_like(fallback)
        emergency = emergency.at[:, self.spec.nu_slice].set(raw_f_min)
        # Maximal diagonal impedance retracts the probe toward the nominal
        # surface pose before/while crossing a soft-to-hard material boundary.
        # Off-diagonal chart entries remain zero, so the emergency stiffness is
        # SPD and axis-aligned rather than an arbitrary sampled coupling.
        diag = jnp.asarray([0, 3, 5]) + self.spec.s_slice.start
        emergency = emergency.at[:, diag].set(1.0)
        recovery_scan = jnp.clip(
            cfg.scan_rate / max(cfg.coord_scale, 1.0e-6), -1.0, 1.0
        )
        emergency = emergency.at[1:, self.spec.r_slice.start].set(
            recovery_scan
        )
        return emergency

    def emergency_plan_is_active(self, plan):
        cfg = self._config
        lo = cfg.f_min - cfg.f_cmd_pad
        hi = cfg.f_max + cfg.f_cmd_pad
        raw_f_min = 2.0 * (cfg.f_min - lo) / max(hi - lo, 1.0e-6) - 1.0
        first = plan[0]
        stiffness = first[self.spec.s_slice]
        diag = stiffness[jnp.asarray([0, 3, 5])]
        offdiag = stiffness[jnp.asarray([1, 2, 4])]
        # The shifted recovery tail deliberately moves at scan_rate, so path
        # motion cannot distinguish it from a performance plan.  Its
        # zero-force/max-diagonal signature remains unique and keeps it marked
        # emergency-derived until a revalidated refined plan takes control.
        return (
            first[self.spec.nu_slice][0] <= raw_f_min + 1.0e-6
        ) & (
            jnp.min(diag) >= 1.0 - 1.0e-6
        ) & (
            jnp.max(jnp.abs(offdiag)) <= 1.0e-6
        )

    def emergency_plan_should_override(self, state):
        xi, eta = state.info["xi"], state.info["eta"]
        force = self._contact_force_at(state.pipeline_state, xi, eta)
        return (force < self._config.f_min) | (force > self._config.f_max)

    def safety_index(self, state) -> jax.Array:
        """Normalized realized-state safety index used by ISSA/AdamBA.

        ``phi <= 0`` denotes the task safe set.  This is intentionally a pure
        state contract: an ISSA candidate must first pass through the true MJX
        ``step`` before it can be classified.  Path tracking is performance,
        not safety, and is therefore excluded.  The three safety modes are
        realized force bounds, contact retention, and deformation.
        """
        cfg = self._config
        xi, eta = state.info["xi"], state.info["eta"]
        force = self._contact_force_at(state.pipeline_state, xi, eta)
        deformation = self._penetration_at(state.pipeline_state, xi, eta)
        force_scale = max(cfg.f_max - cfg.f_min, 1.0)
        deformation_scale = max(cfg.deformation_scale, 1.0e-6)
        contact_threshold = 0.5
        return jnp.max(jnp.asarray([
            (cfg.f_min - force) / force_scale,
            (force - cfg.f_max) / force_scale,
            (contact_threshold - force) / contact_threshold,
            (deformation - cfg.deformation_safe) / deformation_scale,
        ], dtype=jnp.float32))

    # --- MGA soft-feasibility: h_surf (eq 727) + h_normal (eq 731) + g_force ---
    def constraint_residual(self, state, action, ctx=None):
        cfg = self._config
        ps = state.pipeline_state
        xi, eta = state.info["xi"], state.info["eta"]
        _, n_s, p_d, _ = self._desired_pose(xi, eta)
        p_h = ps.site_xpos[self._ee_site]
        R_h = ps.site_xmat[self._ee_site].reshape(3, 3)
        h_surf = p_h - p_d                                  # on-surface (eq 727)
        if cfg.soft_contact_manifold:                       # soft-contact manifold: aim at the DEFORMED
            F_real = self._contact_force_at(ps, xi, eta)    # contact S_0 − δ*·n; δ* estimated realized
            delta_real = self._penetration_at(ps, xi, eta)
            k_hat = F_real / (delta_real + cfg.eps_delta)   # k̂ = F_real / (δ_real + ε_δ)
            delta_star = jnp.clip(cfg.f_target / (k_hat + cfg.eps_k), 0.0, cfg.delta_max)  # δ̂*, 0≤δ*≤δ_max
            delta_star = jnp.where(delta_real > cfg.eps_delta, delta_star, 0.0)   # only IN contact (k̂ valid)
            h_surf = h_surf + delta_star * n_s              # C_soft = p_h − p_d + δ̂*·n  (eq)
        elif cfg.h_surf_tangential:                         # #1 (crude): drop the normal penetration
            h_surf = h_surf - jnp.dot(h_surf, n_s) * n_s    # AL scores only the TANGENTIAL path
        h_normal = R_h[:, 2] + n_s                          # z aligned to -n_s (eq 731)
        h = jnp.concatenate([h_surf, h_normal])             # 6 equalities
        F_n_cmd = self._force_cmd(action[self.spec.nu_slice][0])
        g = jnp.array([F_n_cmd - cfg.f_max, cfg.f_min - F_n_cmd])   # F_min ≤ F_n ≤ F_max
        return h, g


class SurfaceScanDomainEnv:
    """Reset-key randomized family of shape-compatible scan environments.

    Brax vectorization supplies a different reset key to each environment.  The
    key selects one statically constructed domain, and ``lax.switch`` dispatches
    reset/step without exposing that index in the policy observation. In Brax
    training this yields a heterogeneous parallel batch; AutoReset may retain an
    environment's selected domain across its episodes. Keeping this wrapper in
    the task module makes material/geometry randomization an environment concern
    rather than hard-coded logic in the RL trainer.
    """

    def __init__(self, domains):
        self.domains = tuple(domains)
        if not self.domains:
            raise ValueError("SurfaceScanDomainEnv needs at least one domain")
        action_sizes = {int(env.action_size) for env in self.domains}
        observation_sizes = {int(env.observation_size) for env in self.domains}
        if len(action_sizes) != 1 or len(observation_sizes) != 1:
            raise ValueError(
                "all randomized Panda domains must share action/observation sizes"
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
        return state.replace(
            info={**state.info, "_rl_domain_index": domain_index}
        )

    def step(self, state, action):
        domain_index = state.info["_rl_domain_index"]
        branches = tuple(
            (lambda s, env=env: env.step(s, action)) for env in self.domains
        )
        return jax.lax.switch(domain_index, branches, state)


class SurfaceScanResidualActionEnv:
    """Train a residual policy around a fixed, executable scan primitive."""

    def __init__(self, env, action_bias, action_scale=None):
        self.env = env
        self.action_bias = jnp.asarray(action_bias, dtype=jnp.float32)
        if self.action_bias.shape != (int(env.action_size),):
            raise ValueError(
                "action_bias must match the surface-scan primitive: "
                f"{self.action_bias.shape} != {(int(env.action_size),)}"
            )
        self.action_scale = jnp.asarray(
            jnp.ones_like(self.action_bias) if action_scale is None else action_scale,
            dtype=jnp.float32,
        )
        if self.action_scale.shape != self.action_bias.shape:
            raise ValueError(
                "action_scale must match action_bias: "
                f"{self.action_scale.shape} != {self.action_bias.shape}"
            )
        if bool(jnp.any(self.action_scale <= 0.0)):
            raise ValueError("action_scale entries must be positive")

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
            self.action_scale * jnp.asarray(residual_action) + self.action_bias,
            -1.0,
            1.0,
        )
        return self.env.step(state, action)


# Compatibility names preserve every existing solver/config import while the
# canonical API is task-named and accepts ``config.robot``.
PandaSurfaceScanConfig = SurfaceScanConfig
PandaSurfaceScanEnv = SurfaceScanEnv
PandaSurfaceScanDomainEnv = SurfaceScanDomainEnv
PandaResidualActionEnv = SurfaceScanResidualActionEnv


__all__ = [
    "SurfaceScanConfig",
    "SurfaceScanEnv",
    "SurfaceScanDomainEnv",
    "SurfaceScanResidualActionEnv",
    "PandaSurfaceScanConfig",
    "PandaSurfaceScanEnv",
    "PandaSurfaceScanDomainEnv",
    "PandaResidualActionEnv",
]
