"""Single-arm contact-rich peg insertion on MuJoCo/MJX.

The robot, tool, fixture, low-level impedance controller, and planning task stay
separate: a :class:`RobotProfile` supplies the arm, :class:`MjcfSceneComposer`
attaches one keyed peg and a convex-decomposed fixed socket, and this module owns
only insertion state, reward, constraints, reliability, and OOD semantics.

The normalized 13-D primitive is

``[dpos_hole(3), drot_hole(3), svec(log K_translation)(6), axial_force(1)]``.

All planners see the same primitive and the same impedance servo.  A nominal
planning environment and a shape-compatible randomized execution environment
can be paired without exposing the execution parameters to observations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp

try:
    import mujoco
    from mujoco import mjx
    from mujoco.mjx._src import support as _mjx_support
    from brax.envs.base import PipelineEnv, State
    from brax.io import mjcf
    BRAX_AVAILABLE = True
except Exception:  # pragma: no cover - optional backend
    BRAX_AVAILABLE = False
    PipelineEnv = object
    State = object

from genedynamics.core.control.cartesian_impedance import (
    end_effector_kinematics,
    map_cartesian_wrench,
    orientation_error,
)
from genedynamics.core.control.stiffness import PrimitiveSpec, stiffness_log_to_pd
from genedynamics.envs.composition import MjcfSceneComposer, PegToolSpec, SocketSpec
from genedynamics.robots import RobotBinding, get_robot_registry
from genedynamics.robots.profile import (
    CARTESIAN_JACOBIAN,
    FIXED_BASE,
    SINGLE_TOOL,
    TORQUE_CONTROL,
)


_STIFF_D = 3
_QUEUE_CAPACITY = 3  # zero, one, or two control/sensor steps; topology stays fixed


def _skew(v):
    x, y, z = v
    return jnp.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _rotation_exp(rotvec):
    """Stable SO(3) exponential for small insertion-angle increments."""
    theta2 = jnp.sum(rotvec * rotvec)
    theta = jnp.sqrt(theta2 + 1.0e-16)
    a = jnp.where(theta2 < 1.0e-8, 1.0 - theta2 / 6.0, jnp.sin(theta) / theta)
    b = jnp.where(
        theta2 < 1.0e-8,
        0.5 - theta2 / 24.0,
        (1.0 - jnp.cos(theta)) / theta2,
    )
    K = _skew(rotvec)
    return jnp.eye(3, dtype=rotvec.dtype) + a * K + b * (K @ K)


def _np_rotation_xyz(angles: Sequence[float]) -> np.ndarray:
    rx, ry, rz = (float(x) for x in angles)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float64)
    Ry = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float64)
    Rz = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float64)
    return Rz @ Ry @ Rx


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(R, np.float64).reshape(-1))
    return quat


def _home_tool_pose(profile) -> Tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(profile.model_path())
    data = mujoco.MjData(model)
    home = np.asarray(profile.controller_defaults["home_qpos"], np.float64)
    data.qpos[: home.size] = home
    mujoco.mj_forward(model, data)
    name = profile.elements["tool_mount"].name
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return np.asarray(data.site_xpos[sid]), np.asarray(data.site_xmat[sid]).reshape(3, 3)


@dataclass
class PegInsertConfig:
    robot: str = "panda"
    tool: str = "rectangular_peg"
    level: str = "wide"  # wide | tight | ood
    domain_seed: int = 0
    dt: float = 0.02
    timestep: float = 0.002
    solver_iterations: int = 20
    solver_ls_iterations: int = 20

    peg_half_size: Tuple[float, float] = (0.010, 0.006)
    peg_length: float = 0.050
    peg_mass: float = 0.08
    socket_depth: float = 0.040
    clearance: float = 0.0015  # per side
    wall_thickness: float = 0.010
    bottom_thickness: float = 0.006
    chamfer_depth: float = 0.006
    chamfer_width: float = 0.002
    chamfer_steps: int = 2
    approach_gap: float = 0.008
    hole_position_offset: Tuple[float, float] = (0.0, 0.0)
    hole_orientation_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    friction: float = 0.6
    torsional_friction: float = 0.005
    rolling_friction: float = 0.0001
    contact_solref: str = "0.03 1"
    contact_solimp: str = "0.9 0.95 0.001"
    ood_clearance_range: Tuple[float, float] = (0.0005, 0.0012)
    ood_friction_range: Tuple[float, float] = (0.25, 0.9)
    ood_position_range: float = 0.003
    ood_angle_range: float = 0.05236  # 3 degrees
    ood_solref_time_range: Tuple[float, float] = (0.006, 0.025)

    translation_step: Tuple[float, float, float] = (0.0010, 0.0010, 0.0015)
    rotation_step: float = 0.015
    insertion_rate: float = 0.0015
    s_ref_diag: float = 6.0
    s_scale: float = 1.2
    stiffness_mode: str = "log_spd"
    rotational_stiffness: float = 60.0
    translational_damping: float = 18.0
    rotational_damping: float = 5.0
    gravity_compensation: float = 1.0
    fast_impedance_loop: bool = True

    f_target: float = 10.0
    f_min: float = 0.0
    f_max: float = 30.0
    f_cmd_pad: float = 10.0
    kp_force: float = 0.25
    ki_force: float = 6.0
    force_int_max: float = 15.0
    lateral_force_limit: float = 20.0
    bending_torque_limit: float = 1.5
    torsional_torque_limit: float = 1.0

    action_delay_steps: int = 0
    sensor_delay_steps: int = 0
    force_sensor_bias: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    torque_sensor_bias: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    contact_switch_volatility_threshold: float = 0.25
    contact_switch_axial_action_limit: float = 0.5

    success_depth: float = 0.032
    success_lateral_tol: float = 0.0012
    success_angle_tol: float = 0.060
    success_hold_steps: int = 3
    jam_progress_tol: float = 0.00008
    jam_dwell_steps: int = 4
    jam_force_fraction: float = 0.75
    contact_threshold: float = 0.5

    w_depth: float = 5.0
    w_lateral: float = 1.5
    w_orientation: float = 0.8
    w_force: float = 0.25
    w_safety: float = 8.0
    w_jam: float = 2.0
    w_action: float = 0.002
    w_stiffness: float = 0.002

    # Robust tightening used only by the model-based soft-feasibility rollout.
    # Reported violations and task success always use the physical limits above.
    soft_safety_margin: float = 1.0
    # Candidate revalidation is a hard safety decision and therefore has a
    # separate margin.  Keeping the optimization buffer in
    # ``soft_safety_margin`` while certifying against the physical set avoids
    # turning every conservative-buffer crossing into an emergency retreat.
    # Learned reliability supplies the additional realization/OOD veto.
    risk_safety_margin: float = 1.0

    geometry_gate_floor: float = 0.02
    geometry_gate_pose_scale: float = 0.003
    geometry_gate_angle_scale: float = 0.05
    geometry_gate_eta_pose: float = 1.0
    geometry_gate_eta_wrench: float = 2.0
    geometry_gate_eta_stall: float = 1.5
    realization_probe_horizon: int = 3
    realization_probe_eps: float = 0.04
    realization_control_reg: float = 1.0e-4
    realization_control_max_action: float = 0.25
    observation_mode: str = "contact"
    clean_manifold_force: bool = True


def _resolved_physics(cfg: PegInsertConfig) -> Dict[str, object]:
    level = str(cfg.level).lower()
    if level not in {"wide", "tight", "ood"}:
        raise ValueError("peg insertion level must be 'wide', 'tight', or 'ood'")
    out: Dict[str, object] = {
        "clearance": float(cfg.clearance),
        "friction": float(cfg.friction),
        "position": np.asarray(cfg.hole_position_offset, np.float64),
        "angles": np.asarray(cfg.hole_orientation_offset, np.float64),
        "solref": str(cfg.contact_solref),
    }
    if level == "wide":
        out["clearance"] = max(float(cfg.clearance), 0.0015)
    elif level == "tight":
        out["clearance"] = min(float(cfg.clearance), 0.0008)
    else:
        rng = np.random.default_rng(int(cfg.domain_seed) + 4109)
        out["clearance"] = float(rng.uniform(*cfg.ood_clearance_range))
        out["friction"] = float(rng.uniform(*cfg.ood_friction_range))
        out["position"] = rng.uniform(
            -cfg.ood_position_range, cfg.ood_position_range, size=2
        )
        out["angles"] = rng.uniform(
            -cfg.ood_angle_range, cfg.ood_angle_range, size=3
        )
        time_constant = float(rng.uniform(*cfg.ood_solref_time_range))
        out["solref"] = f"{time_constant:.7g} 1"
    return out


def _build_model(profile, cfg: PegInsertConfig, physics: Dict[str, object]):
    root_pos, root_rot = _home_tool_pose(profile)
    peg = PegToolSpec(
        tool_id=cfg.tool,
        half_size_x=float(cfg.peg_half_size[0]),
        half_size_y=float(cfg.peg_half_size[1]),
        length=float(cfg.peg_length),
        mass=float(cfg.peg_mass),
    )
    clearance = float(physics["clearance"])
    socket = SocketSpec(
        hole_half_size_x=peg.half_size_x + clearance,
        hole_half_size_y=peg.half_size_y + clearance,
        depth=float(cfg.socket_depth),
        wall_thickness=float(cfg.wall_thickness),
        bottom_thickness=float(cfg.bottom_thickness),
        chamfer_depth=float(cfg.chamfer_depth),
        chamfer_width=float(cfg.chamfer_width),
        chamfer_steps=int(cfg.chamfer_steps),
    )
    hole_rot = root_rot @ _np_rotation_xyz(physics["angles"])
    entrance = (
        root_pos
        + root_rot[:, 2] * float(cfg.peg_length)
        + root_rot @ np.asarray(
            [physics["position"][0], physics["position"][1], cfg.approach_gap],
            np.float64,
        )
    )
    friction = (
        float(physics["friction"]),
        float(cfg.torsional_friction),
        float(cfg.rolling_friction),
    )
    composer = MjcfSceneComposer(profile)
    if profile.controller_defaults.get("strip_mesh_geoms", False):
        composer.strip_mesh_geoms()
    if profile.controller_defaults.get("replace_actuators_with_motors", False):
        composer.replace_actuators_with_motors(
            profile.controller_defaults["torque_limits"]
        )
    composer.add_peg_tool(
        peg, friction=friction, solref=physics["solref"], solimp=cfg.contact_solimp
    )
    composer.add_rectangular_socket(
        socket,
        position=entrance,
        quaternion=_matrix_to_quat(hole_rot),
        friction=friction,
        solref=physics["solref"],
        solimp=cfg.contact_solimp,
    )
    model = composer.compile()
    model.opt.timestep = float(cfg.timestep)
    model.opt.iterations = int(cfg.solver_iterations)
    model.opt.ls_iterations = int(cfg.solver_ls_iterations)
    return model, peg, socket, entrance, hole_rot


class PegInsertEnv(PipelineEnv):
    """Robot-parameterized, fixed-socket insertion task."""

    def __init__(self, config: Optional[PegInsertConfig] = None, **kwargs):
        if not BRAX_AVAILABLE:
            raise ImportError("brax + mujoco(mjx) are required for PegInsertEnv")
        cfg = config or PegInsertConfig(**kwargs)
        if not 0 <= int(cfg.action_delay_steps) < _QUEUE_CAPACITY:
            raise ValueError("action_delay_steps must be 0, 1, or 2")
        if not 0 <= int(cfg.sensor_delay_steps) < _QUEUE_CAPACITY:
            raise ValueError("sensor_delay_steps must be 0, 1, or 2")
        self._config = cfg
        self._robot_profile = get_robot_registry().get_profile(
            "manipulator", str(cfg.robot).lower()
        )
        if self._robot_profile is None:
            raise ValueError(f"unknown manipulator robot profile: {cfg.robot}")
        self._robot_profile.require(
            {FIXED_BASE, CARTESIAN_JACOBIAN, TORQUE_CONTROL, SINGLE_TOOL}
        )
        if cfg.tool != "rectangular_peg":
            raise ValueError(f"unsupported insertion tool: {cfg.tool}")

        self._physics = _resolved_physics(cfg)
        mj, self._peg_spec, self._socket_spec, entrance, hole_rot = _build_model(
            self._robot_profile, cfg, self._physics
        )
        sys = mjcf.load_model(mj)
        n_frames = max(1, int(round(cfg.dt / cfg.timestep)))
        super().__init__(sys=sys, backend="mjx", n_frames=n_frames)
        self._mj_model = mj
        self._mjx_model = mjx.put_model(mj)
        self.spec = PrimitiveSpec(pos_dim=6, stiff_dim=_STIFF_D, feed_dim=1)
        self._robot_binding = RobotBinding.from_mujoco_model(self._robot_profile, mj)
        self._home_qpos = jnp.asarray(
            self._robot_profile.controller_defaults["home_qpos"], jnp.float32
        )
        self._root_site = mujoco.mj_name2id(
            mj, mujoco.mjtObj.mjOBJ_SITE, self._peg_spec.root_site_name
        )
        self._tip_site = mujoco.mj_name2id(
            mj, mujoco.mjtObj.mjOBJ_SITE, self._peg_spec.tip_site_name
        )
        self._ee_body = int(self.sys.site_bodyid[self._tip_site])
        self._peg_geom = mujoco.mj_name2id(
            mj, mujoco.mjtObj.mjOBJ_GEOM, self._peg_spec.geom_name
        )
        socket_ids = []
        for gid in range(mj.ngeom):
            name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name.startswith("socket_"):
                socket_ids.append(gid)
        self._socket_geoms = jnp.asarray(socket_ids, jnp.int32)
        self._socket_entrance_site = mujoco.mj_name2id(
            mj, mujoco.mjtObj.mjOBJ_SITE, self._socket_spec.entrance_site_name
        )
        self._socket_pos = jnp.asarray(entrance, jnp.float32)
        self._socket_rot = jnp.asarray(hole_rot, jnp.float32)
        self._socket_axis = self._socket_rot[:, 2]
        self._translation_step = jnp.asarray(cfg.translation_step, jnp.float32)
        diag_idx = jnp.asarray([0, 3, 5], jnp.int32)
        self._s_ref = jnp.zeros((self.spec.stiff_width,), jnp.float32).at[diag_idx].set(
            cfg.s_ref_diag
        )
        self._force_bias = jnp.asarray(cfg.force_sensor_bias, jnp.float32)
        self._torque_bias = jnp.asarray(cfg.torque_sensor_bias, jnp.float32)

    @property
    def action_size(self) -> int:
        return self.spec.total_width

    @property
    def reliability_feature_size(self) -> int:
        return 44

    @property
    def manifold_constraint_size(self) -> int:
        return 7 if self._config.clean_manifold_force else 6

    @property
    def inequality_constraint_size(self) -> int:
        return 4

    def _force_cmd(self, raw):
        cfg = self._config
        lo, hi = cfg.f_min - cfg.f_cmd_pad, cfg.f_max + cfg.f_cmd_pad
        return lo + 0.5 * (jnp.clip(raw, -1.0, 1.0) + 1.0) * (hi - lo)

    def _force_action_raw(self, force):
        """Inverse of ``_force_cmd`` for task-owned executable actions."""
        cfg = self._config
        lo, hi = cfg.f_min - cfg.f_cmd_pad, cfg.f_max + cfg.f_cmd_pad
        return jnp.clip(2.0 * (force - lo) / (hi - lo) - 1.0, -1.0, 1.0)

    def _stiffness(self, svec):
        mode = self._config.stiffness_mode
        if mode == "fixed":
            return stiffness_log_to_pd(self._s_ref, _STIFF_D)
        if mode == "none":
            return jnp.eye(3, dtype=jnp.float32)
        if mode == "euclid":
            from genedynamics.core.control.stiffness import svec2sym
            return jnp.diag(jax.nn.softplus(jnp.diagonal(svec2sym(svec, 3))))
        return stiffness_log_to_pd(svec, _STIFF_D)

    def _unpack(self, action):
        raw_pose = action[self.spec.r_slice]
        raw_s = action[self.spec.s_slice]
        raw_f = action[self.spec.nu_slice][0]
        delta = jnp.concatenate(
            [self._translation_step * raw_pose[:3], self._config.rotation_step * raw_pose[3:]]
        )
        return delta, self._s_ref + self._config.s_scale * raw_s, self._force_cmd(raw_f)

    def _command_target(self, command_pose):
        p_tip = self._socket_pos + self._socket_rot @ command_pose[:3]
        R = self._socket_rot @ _rotation_exp(command_pose[3:])
        p_root = p_tip - R[:, 2] * self._peg_spec.length
        return p_root, R

    def _actual_pose(self, ps):
        tip = ps.site_xpos[self._tip_site]
        R = ps.site_xmat[self._tip_site].reshape(3, 3)
        local_pos = self._socket_rot.T @ (tip - self._socket_pos)
        local_rot_error = self._socket_rot.T @ orientation_error(R, self._socket_rot)
        return local_pos, local_rot_error

    def _contact_wrench(self, ps):
        contact = ps.contact
        if contact is None or contact.dist.shape[0] == 0:
            return jnp.zeros((6,), jnp.float32), jnp.int32(0), jnp.float32(0.0)
        count = contact.dist.shape[0]
        root = ps.site_xpos[self._root_site]
        force = jnp.zeros((3,), jnp.float32)
        moment = jnp.zeros((3,), jnp.float32)
        active_count = jnp.int32(0)
        penetration = jnp.float32(0.0)
        for index in range(count):
            pair = contact.geom[index]
            on_peg = (pair[0] == self._peg_geom) | (pair[1] == self._peg_geom)
            on_socket = jnp.any(
                (self._socket_geoms == pair[0]) | (self._socket_geoms == pair[1])
            )
            active = on_peg & on_socket & (contact.dist[index] < 0.0)
            local = _mjx_support.contact_force(self._mjx_model, ps, index)[:3]
            frame = contact.frame[index].reshape(3, 3)
            world_on_geom2 = frame.T @ local
            sign = jnp.where(pair[1] == self._peg_geom, 1.0, -1.0)
            fi = sign * world_on_geom2
            fi = jnp.where(active, fi, jnp.zeros_like(fi))
            force = force + fi
            moment = moment + jnp.cross(contact.pos[index] - root, fi)
            active_count = active_count + active.astype(jnp.int32)
            penetration = jnp.maximum(
                penetration, jnp.where(active, -contact.dist[index], 0.0)
            )
        return jnp.concatenate([force, moment]), active_count, penetration

    def wrench_components(self, wrench):
        local = jnp.concatenate(
            [self._socket_rot.T @ wrench[:3], self._socket_rot.T @ wrench[3:]]
        )
        return {
            "local": local,
            "lateral_force": jnp.linalg.norm(local[:2]),
            "axial_force": jnp.abs(local[2]),
            "bending_torque": jnp.linalg.norm(local[3:5]),
            "torsional_torque": jnp.abs(local[5]),
        }

    def reset(self, rng: jax.Array) -> State:
        ps = self.pipeline_init(self._home_qpos, jnp.zeros(self.sys.qd_size()))
        initial_pose = jnp.asarray([0.0, 0.0, -self._config.approach_gap, 0.0, 0.0, 0.0])
        actual_pose, actual_angle = self._actual_pose(ps)
        wrench, contact_count, penetration = self._contact_wrench(ps)
        # The force channel is affine, so an all-zero action decodes to the
        # midpoint of the padded force range (15 N for the canonical task),
        # not to zero force.  Delayed execution must begin with a genuinely
        # neutral command; otherwise reset silently preloads the force
        # integrator before the planner's first action reaches the robot.
        neutral_action = jnp.zeros((self.action_size,), jnp.float32)
        neutral_action = neutral_action.at[self.spec.nu_slice].set(
            self._force_action_raw(self._config.f_min)
        )
        info = {
            "command_pose": initial_pose,
            "reliability_pose": actual_pose,
            "reliability_angle": actual_angle,
            "reliability_prev_pose": actual_pose,
            "reliability_prev_angle": actual_angle,
            "prev_s": self._s_ref,
            "prev_action": jnp.zeros((self.action_size,), jnp.float32),
            "action_queue": jnp.broadcast_to(
                neutral_action, (_QUEUE_CAPACITY, self.action_size)
            ),
            "wrench_queue": jnp.zeros((_QUEUE_CAPACITY, 6), jnp.float32),
            # Execution metadata travels with the measured state.  A nominal
            # planning model may intentionally omit OOD delay, but task-owned
            # projection/revalidation must still honor the real committed
            # queue before declaring a candidate safe.
            "execution_action_delay_steps": jnp.int32(
                self._config.action_delay_steps
            ),
            "execution_sensor_delay_steps": jnp.int32(
                self._config.sensor_delay_steps
            ),
            "measured_wrench": wrench,
            "true_wrench": wrench,
            "contact_count": contact_count,
            "contact_count_delta": jnp.float32(0.0),
            "contact_volatility": jnp.float32(0.0),
            "penetration": penetration,
            "force_int": jnp.float32(0.0),
            "step": jnp.int32(0),
            "prev_depth": jnp.float32(0.0),
            "max_depth": jnp.float32(0.0),
            "stall_steps": jnp.int32(0),
            "jammed": jnp.float32(0.0),
            "jammed_once": jnp.float32(0.0),
            "recovered": jnp.float32(0.0),
            "success_hold": jnp.int32(0),
            "success": jnp.float32(0.0),
            "contact_mode": jnp.int32(0),
        }
        obs = self._get_obs(ps, info)
        zero = jnp.float32(0.0)
        return State(
            ps,
            obs,
            zero,
            zero,
            {
                "reward": zero,
                "insertion_depth": zero,
                "lateral_force": zero,
                "axial_force": zero,
                "bending_torque": zero,
                "jammed": zero,
                "success": zero,
            },
            info,
        )

    def _impedance_tau(self, ps, command_pose, svec, force_cmd, force_int, measured):
        cfg = self._config
        p_des, R_des = self._command_target(command_pose)
        p, R, jacp, jacr, lin_v, ang_v = end_effector_kinematics(
            self.sys, ps, self._root_site, self._ee_body
        )
        K = self._stiffness(svec)
        comp = self.wrench_components(measured)
        axial_measured = comp["axial_force"]
        effective_force = jnp.clip(
            force_cmd + cfg.kp_force * (force_cmd - axial_measured) + force_int,
            cfg.f_min,
            cfg.f_max + cfg.f_cmd_pad,
        )
        force = (
            K @ (p_des - p)
            - cfg.translational_damping * lin_v
            + effective_force * self._socket_axis
        )
        moment = (
            cfg.rotational_stiffness * orientation_error(R, R_des)
            - cfg.rotational_damping * ang_v
        )
        dofs = jnp.asarray(self._robot_binding.dof_indices)
        return map_cartesian_wrench(
            jacp,
            jacr,
            force,
            moment,
            ps.qfrc_bias[dofs],
            gravity_compensation=cfg.gravity_compensation,
            torque_limits=self._robot_profile.controller_defaults["torque_limits"],
        )

    def step(self, state: State, action: jax.Array) -> State:
        cfg = self._config
        clipped = jnp.clip(action, -1.0, 1.0)
        action_queue = jnp.concatenate(
            [clipped[None, :], state.info["action_queue"][:-1]], axis=0
        )
        applied = action_queue[int(cfg.action_delay_steps)]
        delta, svec, force_raw = self._unpack(applied)
        command_pose = state.info["command_pose"] + delta
        # Keep commands in a physically meaningful local workspace.
        command_pose = command_pose.at[:2].set(jnp.clip(command_pose[:2], -0.012, 0.012))
        command_pose = command_pose.at[2].set(
            jnp.clip(command_pose[2], -cfg.approach_gap, cfg.socket_depth + 0.002)
        )
        command_pose = command_pose.at[3:].set(jnp.clip(command_pose[3:], -0.18, 0.18))
        force_cmd = jnp.clip(force_raw, cfg.f_min, cfg.f_max)

        if cfg.fast_impedance_loop:
            def substep(carry, _):
                ps_i, integral, measured_i = carry
                tau = self._impedance_tau(
                    ps_i, command_pose, svec, force_cmd, integral, measured_i
                )
                ps_i = self._pipeline.step(self.sys, ps_i, tau, self._debug)
                true_i, _, _ = self._contact_wrench(ps_i)
                measured_i = true_i + jnp.concatenate([self._force_bias, self._torque_bias])
                axial = self.wrench_components(measured_i)["axial_force"]
                integral = jnp.clip(
                    integral + cfg.ki_force * cfg.timestep * (force_cmd - axial),
                    -cfg.force_int_max,
                    cfg.force_int_max,
                )
                return (ps_i, integral, measured_i), None

            (ps, force_int, _), _ = jax.lax.scan(
                substep,
                (state.pipeline_state, state.info["force_int"], state.info["measured_wrench"]),
                (),
                self._n_frames,
            )
        else:
            tau = self._impedance_tau(
                state.pipeline_state,
                command_pose,
                svec,
                force_cmd,
                state.info["force_int"],
                state.info["measured_wrench"],
            )
            ps = self.pipeline_step(state.pipeline_state, tau)
            true_now, _, _ = self._contact_wrench(ps)
            axial = self.wrench_components(true_now)["axial_force"]
            force_int = jnp.clip(
                state.info["force_int"] + cfg.ki_force * cfg.dt * (force_cmd - axial),
                -cfg.force_int_max,
                cfg.force_int_max,
            )

        true_wrench, contact_count, penetration = self._contact_wrench(ps)
        contact_count_delta = (
            contact_count - state.info["contact_count"]
        ).astype(jnp.float32)
        contact_jump = jnp.clip(
            jnp.abs(contact_count_delta) / 4.0,
            0.0,
            1.0,
        )
        contact_volatility = jnp.maximum(
            0.5 * state.info["contact_volatility"], contact_jump
        )
        wrench_queue = jnp.concatenate(
            [true_wrench[None, :], state.info["wrench_queue"][:-1]], axis=0
        )
        measured_wrench = (
            wrench_queue[int(cfg.sensor_delay_steps)]
            + jnp.concatenate([self._force_bias, self._torque_bias])
        )
        actual_pose, actual_angle = self._actual_pose(ps)
        depth = jnp.clip(actual_pose[2], 0.0, cfg.socket_depth)
        progress = depth - state.info["prev_depth"]
        components = self.wrench_components(true_wrench)
        in_contact = components["axial_force"] > cfg.contact_threshold
        high_transverse_wrench = (
            components["lateral_force"]
            > cfg.jam_force_fraction * cfg.lateral_force_limit
        ) | (
            components["bending_torque"]
            > cfg.jam_force_fraction * cfg.bending_torque_limit
        )
        # A centered peg can jam axially without generating a large lateral
        # wrench.  In that mode the signature is sustained contact load at the
        # requested insertion force with no depth progress.  This is a task
        # state transition, not a physical force-limit violation: the latter
        # remains governed independently by ``f_max`` below and in the formal
        # metrics.
        high_axial_load = components["axial_force"] > jnp.maximum(
            cfg.contact_threshold,
            cfg.jam_force_fraction * cfg.f_target,
        )
        high_wrench = high_transverse_wrench | high_axial_load
        # Jamming is failure to progress while the *applied* delayed command
        # is still trying to insert.  Residual contact force can take several
        # impedance intervals to decay after a zero-force/retract emergency;
        # counting that commanded unloading interval as continued insertion
        # stall makes a timely safety response indistinguishable from a jam.
        # Physical force/torque limits remain evaluated from the true wrench
        # independently, so this intent condition cannot hide an unsafe load.
        insertion_intent = (
            (delta[2] > 1.0e-8)
            | (force_cmd > cfg.f_min + 1.0e-6)
        )
        stalled = (
            insertion_intent
            & in_contact
            & (jnp.abs(progress) < cfg.jam_progress_tol)
            & high_wrench
        )
        stall_steps = jnp.where(stalled, state.info["stall_steps"] + 1, 0)
        jammed = (stall_steps >= cfg.jam_dwell_steps).astype(jnp.float32)
        jammed_once = jnp.maximum(state.info["jammed_once"], jammed)
        recovered_now = (
            (state.info["jammed_once"] > 0.5)
            & (jammed < 0.5)
            & (progress > cfg.jam_progress_tol)
            # Normal insertion progress commonly resumes under the desired
            # axial force.  Requiring that force to disappear would make an
            # actual contact recovery impossible to record.
            & (~high_transverse_wrench)
            & (components["axial_force"] <= cfg.f_max)
        ).astype(jnp.float32)
        recovered = jnp.maximum(state.info["recovered"], recovered_now)
        lateral = jnp.linalg.norm(actual_pose[:2])
        angle = jnp.linalg.norm(actual_angle)
        success_now = (
            (depth >= cfg.success_depth)
            & (lateral <= cfg.success_lateral_tol)
            & (angle <= cfg.success_angle_tol)
            & (components["lateral_force"] <= cfg.lateral_force_limit)
            & (components["axial_force"] <= cfg.f_max)
            & (components["bending_torque"] <= cfg.bending_torque_limit)
            & (components["torsional_torque"] <= cfg.torsional_torque_limit)
        )
        success_hold = jnp.where(success_now, state.info["success_hold"] + 1, 0)
        success = jnp.maximum(
            state.info["success"],
            (success_hold >= cfg.success_hold_steps).astype(jnp.float32),
        )
        mode = jnp.where(
            success > 0.5,
            4,
            jnp.where(jammed > 0.5, 3, jnp.where(depth > 0.002, 2, jnp.where(in_contact, 1, 0))),
        ).astype(jnp.int32)
        info = {
            **state.info,
            "command_pose": command_pose,
            "reliability_pose": actual_pose,
            "reliability_angle": actual_angle,
            "reliability_prev_pose": state.info["reliability_pose"],
            "reliability_prev_angle": state.info["reliability_angle"],
            "prev_s": svec,
            "prev_action": clipped,
            "action_queue": action_queue,
            "wrench_queue": wrench_queue,
            "measured_wrench": measured_wrench,
            "true_wrench": true_wrench,
            "contact_count": contact_count,
            "contact_count_delta": contact_count_delta,
            "contact_volatility": contact_volatility,
            "penetration": penetration,
            "force_int": force_int,
            "step": state.info["step"] + 1,
            "prev_depth": depth,
            "max_depth": jnp.maximum(state.info["max_depth"], depth),
            "stall_steps": stall_steps,
            "jammed": jammed,
            "jammed_once": jammed_once,
            "recovered": recovered,
            "success_hold": success_hold,
            "success": success,
            "contact_mode": mode,
        }
        reward = self._reward(info, actual_pose, actual_angle, components, applied)
        obs = self._get_obs(ps, info)
        return state.replace(
            pipeline_state=ps,
            obs=obs,
            reward=reward,
            done=success,
            metrics={
                "reward": reward,
                "insertion_depth": depth,
                "lateral_force": components["lateral_force"],
                "axial_force": components["axial_force"],
                "bending_torque": components["bending_torque"],
                "jammed": jammed,
                "success": success,
            },
            info=info,
        )

    def _reward(self, info, pose, angle, wrench, action):
        cfg = self._config
        depth_error = (
            cfg.socket_depth
            - jnp.clip(pose[2], -cfg.approach_gap, cfg.socket_depth)
        ) / (cfg.socket_depth + cfg.approach_gap)
        lateral_error = jnp.linalg.norm(pose[:2])
        angle_error = jnp.linalg.norm(angle)
        lateral = lateral_error / max(cfg.clearance + 0.003, 1.0e-6)
        orientation = angle_error / 0.1
        contact_weight = jax.nn.sigmoid(pose[2] / 0.001)
        # In free space, depth is the approach objective.  Once the peg reaches
        # the socket, however, pushing deeper before it lies inside the
        # chamfer's capture basin creates exactly the low-force local minimum
        # that an insertion controller must avoid.  Use the task geometry to
        # continuously hand priority from alignment back to depth; this keeps
        # the objective differentiable and does not introduce a solver-specific
        # mode switch or a tuned per-suite threshold.
        capture_lateral = max(
            cfg.clearance + cfg.chamfer_width,
            cfg.success_lateral_tol,
            1.0e-6,
        )
        capture_angle = max(cfg.success_angle_tol, 1.0e-6)
        alignment_gate = jnp.exp(
            -(lateral_error / capture_lateral) ** 2
            - (angle_error / capture_angle) ** 2
        )
        depth_weight = (1.0 - contact_weight) + contact_weight * (
            0.1 + 0.9 * alignment_gate
        )
        force_error = (
            contact_weight
            * jnp.abs(wrench["axial_force"] - cfg.f_target)
            / max(cfg.f_max, 1.0)
        )
        force_excess = jax.nn.relu(wrench["lateral_force"] / cfg.lateral_force_limit - 1.0)
        axial_excess = jax.nn.relu(wrench["axial_force"] / cfg.f_max - 1.0)
        torque_excess = jax.nn.relu(wrench["bending_torque"] / cfg.bending_torque_limit - 1.0)
        safety = force_excess**2 + axial_excess**2 + torque_excess**2
        stiffness_delta = action[self.spec.s_slice]
        cost = (
            cfg.w_depth * depth_weight * depth_error**2
            + cfg.w_lateral * lateral**2
            + cfg.w_orientation * orientation**2
            + cfg.w_force * force_error**2
            + cfg.w_safety * safety
            + cfg.w_jam * info["jammed"]
            + cfg.w_action * jnp.mean(action[:6] ** 2)
            + cfg.w_stiffness * jnp.mean(stiffness_delta**2)
        )
        return -cost + 10.0 * info["success"]

    def _get_obs(self, ps, info):
        pose, angle = self._actual_pose(ps)
        wrench = info["measured_wrench"]
        scale = jnp.asarray(
            [0.012, 0.012, self._config.socket_depth, 0.18, 0.18, 0.18], jnp.float32
        )
        task = jnp.concatenate(
            [
                jnp.concatenate([pose, angle]) / scale,
                info["command_pose"] / scale,
                wrench / jnp.asarray([30.0, 30.0, 40.0, 2.0, 2.0, 1.5]),
                jnp.asarray(
                    [
                        info["contact_count"] / 4.0,
                        info["penetration"] / 0.002,
                        info["stall_steps"] / max(self._config.jam_dwell_steps, 1),
                        info["jammed"],
                        info["success"],
                    ]
                ),
                info["prev_action"],
            ]
        )
        return jnp.concatenate([ps.qpos, ps.qvel, task])

    # ------------------------------------------------------------------
    # MGA clean geometry and realized controllability
    # ------------------------------------------------------------------
    def _target_pose(self, step):
        z = jnp.clip(
            -self._config.approach_gap
            + (jnp.asarray(step, jnp.float32) + 1.0) * self._config.insertion_rate,
            -self._config.approach_gap,
            self._config.socket_depth,
        )
        return jnp.asarray([0.0, 0.0, z, 0.0, 0.0, 0.0], jnp.float32)

    def _clean_residual(self, command_pose, force_cmd, target_pose):
        pose_scale = jnp.asarray(
            [0.010, 0.010, self._config.socket_depth, 0.10, 0.10, 0.10],
            jnp.float32,
        )
        pose = (command_pose - target_pose) / pose_scale
        if not self._config.clean_manifold_force:
            return pose
        # No-contact approach must not be projected onto a fictitious contact
        # force.  The force equality activates smoothly at the socket entrance.
        contact_weight = jax.nn.sigmoid(target_pose[2] / 0.001)
        force = contact_weight * (force_cmd - self._config.f_target) / max(
            self._config.f_max, 1.0
        )
        return jnp.concatenate([pose, force[None]])

    def _manifold_res_node(self, state, action, step):
        delta, _, force_cmd = self._unpack(action)
        command = state.info["command_pose"] + delta
        return self._clean_residual(command, force_cmd, self._target_pose(step))

    def manifold_residual(self, state, nodes):
        step = state.info["step"]
        return jax.vmap(lambda u: self._manifold_res_node(state, u, step))(
            nodes
        ).reshape(-1)

    def manifold_geometry(self, state, nodes, t0):
        step = state.info["step"]

        def objective(u):
            residual = self._manifold_res_node(state, u, step)
            return 0.5 * jnp.sum(residual * residual)

        return jax.vmap(jax.grad(objective))(nodes)

    def _horizon_residual(self, state, actions, t0, target_shift):
        delta_pose = actions[:, self.spec.r_slice] * jnp.concatenate(
            [self._translation_step, jnp.full((3,), self._config.rotation_step)]
        )
        commands = state.info["command_pose"] + jnp.cumsum(delta_pose, axis=0)
        steps = jnp.asarray(t0, jnp.float32) + jnp.arange(
            actions.shape[0], dtype=jnp.float32
        )
        targets = jax.vmap(self._target_pose)(steps) + target_shift
        force = jax.vmap(self._force_cmd)(actions[:, self.spec.nu_slice][:, 0])
        return jax.vmap(self._clean_residual)(commands, force, targets).reshape(-1)

    def manifold_residual_horizon(self, state, actions, t0):
        shift = jnp.zeros((actions.shape[0], 6), actions.dtype)
        return self._horizon_residual(state, actions, t0, shift)

    def _probe_final_state(self, state, actions):
        def body(s, u):
            s2 = self.step(s, u)
            return s2, None

        return jax.lax.scan(body, state, actions)[0]

    def realization_control_jacobian(self, state, dense_actions):
        """Short-horizon true pose response to all six command channels."""
        horizon = min(
            max(int(self._config.realization_probe_horizon), 1),
            int(dense_actions.shape[0]),
        )
        actions = dense_actions[:horizon]
        eps = max(float(self._config.realization_probe_eps), 1.0e-5)
        basis = jnp.eye(6, dtype=dense_actions.dtype) * eps

        def final_pose(delta):
            perturbed = actions.at[:, :6].add(delta[None, :])
            final = self._probe_final_state(state, perturbed)
            pos, angle = self._actual_pose(final.pipeline_state)
            return jnp.concatenate([pos, angle])

        plus = jax.vmap(final_pose)(basis)
        minus = jax.vmap(final_pose)(-basis)
        return jax.lax.stop_gradient(((plus - minus) / (2.0 * eps)).T)

    def prepare_realization_context(
        self, state, dense_actions, *, gate_controllability=False,
    ):
        B = self.realization_control_jacobian(state, dense_actions)
        pose, angle = self._actual_pose(state.pipeline_state)
        actual = jnp.concatenate([pose, angle])
        error = jax.lax.stop_gradient(actual - state.info["command_pose"])
        realization_gate = jax.lax.cond(
            jnp.asarray(gate_controllability),
            lambda _: self.geometry_reliability(state)["action"][:6],
            lambda _: jnp.ones((6,), dtype=B.dtype),
            operand=None,
        )
        return state.replace(
            info={
                **state.info,
                "_mga_realization_B": B,
                "_mga_realization_error": error,
                "_mga_realization_gate": realization_gate,
            }
        )

    def manifold_residual_horizon_realized(self, state, actions, t0):
        pose, angle = self._actual_pose(state.pipeline_state)
        error = jax.lax.stop_gradient(
            jnp.concatenate([pose, angle]) - state.info["command_pose"]
        )
        ramp = jnp.linspace(0.0, 1.0, actions.shape[0], dtype=actions.dtype)[:, None]
        return self._horizon_residual(state, actions, t0, -ramp * error[None, :])

    def manifold_residual_horizon_controllable(self, state, actions, t0):
        B = state.info["_mga_realization_B"]
        error = state.info["_mga_realization_error"]
        reg = max(float(self._config.realization_control_reg), 1.0e-8)
        correction = -B.T @ jnp.linalg.solve(
            B @ B.T + reg * jnp.eye(6, dtype=B.dtype), error
        )
        correction = jnp.clip(
            correction,
            -self._config.realization_control_max_action,
            self._config.realization_control_max_action,
        )
        probe_horizon = min(
            max(int(self._config.realization_probe_horizon), 1),
            int(actions.shape[0]),
        )
        physical_scale = jnp.concatenate(
            [self._translation_step, jnp.full((3,), self._config.rotation_step)]
        )
        command_shift = correction * physical_scale * float(probe_horizon)
        ramp = jnp.clip(
            (jnp.arange(actions.shape[0], dtype=actions.dtype) + 1.0)
            / float(probe_horizon),
            0.0,
            1.0,
        )
        # The reliability gate applies only to this empirical response-map
        # correction.  The clean path/force manifold remains active in the
        # backend even when the local realization estimate is unreliable.
        realization_gate = state.info["_mga_realization_gate"]
        command_shift = command_shift * realization_gate
        return self._horizon_residual(
            state,
            actions,
            t0,
            ramp[:, None] * command_shift[None, :],
        )

    # ------------------------------------------------------------------
    # Reliability, risk, and safety contracts
    # ------------------------------------------------------------------
    def geometry_reliability(self, state):
        cfg = self._config
        pose, angle = self._actual_pose(state.pipeline_state)
        tracking = pose - state.info["command_pose"][:3]
        angle_tracking = angle - state.info["command_pose"][3:]
        comp = self.wrench_components(state.info["measured_wrench"])
        pose_error = jnp.linalg.norm(tracking) / max(cfg.geometry_gate_pose_scale, 1.0e-6)
        angle_error = jnp.linalg.norm(angle_tracking) / max(
            cfg.geometry_gate_angle_scale, 1.0e-6
        )
        lateral_risk = comp["lateral_force"] / max(cfg.lateral_force_limit, 1.0e-6)
        axial_risk = comp["axial_force"] / max(cfg.f_max, 1.0e-6)
        torque_risk = comp["bending_torque"] / max(cfg.bending_torque_limit, 1.0e-6)
        stall_risk = jnp.clip(
            state.info["stall_steps"] / max(cfg.jam_dwell_steps, 1), 0.0, 1.0
        )
        floor = jnp.clip(jnp.asarray(cfg.geometry_gate_floor), 0.0, 1.0)

        def gate(raw):
            return floor + (1.0 - floor) * jnp.exp(-raw)

        g_pose = gate(cfg.geometry_gate_eta_pose * pose_error)
        g_angle = gate(cfg.geometry_gate_eta_pose * angle_error)
        g_wrench = gate(
            cfg.geometry_gate_eta_wrench
            * (jax.nn.relu(lateral_risk - 0.5) + jax.nn.relu(torque_risk - 0.5))
            + cfg.geometry_gate_eta_stall * stall_risk
        )
        g_force = gate(
            cfg.geometry_gate_eta_wrench * jax.nn.relu(axial_risk - 0.7)
            + cfg.geometry_gate_eta_stall * stall_risk
        )
        action_gate = jnp.ones((self.action_size,), jnp.float32)
        action_gate = action_gate.at[0:2].set(g_pose * g_wrench)
        action_gate = action_gate.at[2].set(g_force)
        action_gate = action_gate.at[3:6].set(g_angle * g_wrench)
        action_gate = action_gate.at[self.spec.s_slice].set(jnp.sqrt(g_wrench * g_force))
        action_gate = action_gate.at[self.spec.nu_slice].set(g_force)
        scalar = (g_pose * g_angle * g_wrench * g_force) ** 0.25
        return {
            "action": action_gate,
            "scalar": scalar,
            "path": g_pose,
            "normal": g_angle,
            "stiffness": jnp.sqrt(g_wrench * g_force),
            "force": g_force,
            "wrench": g_wrench,
            "pose_error": pose_error,
            "angle_error": angle_error,
            "lateral_risk": lateral_risk,
            "axial_risk": axial_risk,
            "torque_risk": torque_risk,
            "stall_risk": stall_risk,
        }

    def _reliability_state_features(self, state):
        pose, angle = self._actual_pose(state.pipeline_state)
        comp = self.wrench_components(state.info["measured_wrench"])
        delta_wrench = state.info["wrench_queue"][0] - state.info["wrench_queue"][1]
        command = state.info["command_pose"]
        tracking = (jnp.concatenate([pose, angle]) - command) / jnp.asarray(
            [0.01, 0.01, self._config.socket_depth, 0.1, 0.1, 0.1]
        )
        motion = jnp.concatenate([
            (pose - state.info["reliability_prev_pose"]) / self._translation_step,
            (angle - state.info["reliability_prev_angle"])
            / self._config.rotation_step,
        ])
        queued = state.info["action_queue"][0, :6]
        return jnp.asarray(
            [
                pose[0] / 0.01,
                pose[1] / 0.01,
                pose[2] / max(self._config.socket_depth, 1.0e-6),
                angle[0] / 0.1,
                angle[1] / 0.1,
                angle[2] / 0.1,
                comp["lateral_force"] / max(self._config.lateral_force_limit, 1.0),
                comp["axial_force"] / max(self._config.f_max, 1.0),
                comp["bending_torque"] / max(self._config.bending_torque_limit, 0.1),
                jnp.linalg.norm(delta_wrench[:3]) / 30.0,
                state.info["contact_count"] / 4.0,
                state.info["stall_steps"] / max(self._config.jam_dwell_steps, 1),
                *tracking,
                *motion,
                *queued,
                state.info["execution_action_delay_steps"] / 2.0,
                state.info["execution_sensor_delay_steps"] / 2.0,
            ],
            jnp.float32,
        )

    def reliability_features(self, state, action):
        """Single-action compatibility view of the sequence feature contract."""
        return self.reliability_features_sequence(state, action[None, :])

    def reliability_features_sequence(self, state, actions):
        """Observable features for the actually committed candidate horizon.

        Insertion can cross free-space, rim-contact, and capture modes inside a
        single MPC proposal.  A first-action feature cannot distinguish a
        smooth insertion plan from one that becomes aggressive two controls
        later.  Aggregate only the task-owned commit horizon so the learned
        bound predicts the same risk interval used by revalidation.
        """
        state_features = self._reliability_state_features(state)
        commit_horizon = self.risk_commit_horizon(state, int(actions.shape[0]))
        mask = (
            jnp.arange(actions.shape[0], dtype=jnp.int32) < commit_horizon
        ).astype(actions.dtype)
        count = jnp.maximum(jnp.sum(mask), 1.0)
        masked = actions * mask[:, None]
        previous = jnp.concatenate(
            [state.info["prev_action"][None, :], actions[:-1]], axis=0
        )
        delta = (actions - previous) * mask[:, None]
        cumulative_translation = jnp.cumsum(masked[:, :3], axis=0)
        lateral_command = jnp.max(
            jnp.linalg.norm(cumulative_translation[:, :2], axis=-1) * mask
        )
        axial_command = jnp.sum(masked[:, 2])
        sequence_features = jnp.asarray(
            [
                actions[0, 0],
                actions[0, 1],
                actions[0, 2],
                jnp.linalg.norm(actions[0, 3:6]),
                jnp.sqrt(jnp.mean((actions[0] - state.info["prev_action"]) ** 2)),
                jnp.sqrt(jnp.sum(masked[:, :3] ** 2) / (3.0 * count)),
                jnp.sqrt(jnp.sum(masked[:, 3:6] ** 2) / (3.0 * count)),
                jnp.sqrt(
                    jnp.sum(masked[:, self.spec.s_slice] ** 2)
                    / (float(self.spec.stiff_width) * count)
                ),
                jnp.sqrt(jnp.sum(delta * delta) / (float(self.action_size) * count)),
                lateral_command,
                axial_command,
                jnp.max(jnp.where(mask > 0.0, actions[:, self.spec.nu_slice][:, 0], -1.0)),
            ],
            jnp.float32,
        )
        return jnp.concatenate([state_features, sequence_features])

    def constraint_residual(self, state, action, ctx=None):
        pose, angle = self._actual_pose(state.pipeline_state)
        command = state.info["command_pose"]
        h = (jnp.concatenate([pose, angle]) - command) / jnp.asarray(
            [0.01, 0.01, self._config.socket_depth, 0.1, 0.1, 0.1]
        )
        comp = self.wrench_components(state.info["true_wrench"])
        g = jnp.asarray(
            [
                comp["lateral_force"] / self._config.lateral_force_limit - 1.0,
                comp["axial_force"] / self._config.f_max - 1.0,
                comp["bending_torque"] / self._config.bending_torque_limit - 1.0,
                comp["torsional_torque"] / self._config.torsional_torque_limit - 1.0,
            ]
        )
        return h, g

    def soft_feasibility_residual(self, state, action, ctx=None):
        """Hard safety residual used by the augmented rollout.

        ``constraint_residual`` retains the full tracking equality for generic
        constraint-aware controllers.  MGA already handles the insertion
        manifold through tangent geometry and realized controllability, so
        charging the AL for transient actual-to-command lag double-counts that
        geometry and makes progress look infeasible.  Soft feasibility is
        therefore reserved for the four physical force/torque limits.
        """
        comp = self.wrench_components(state.info["true_wrench"])
        margin = jnp.clip(
            jnp.asarray(self._config.soft_safety_margin, jnp.float32),
            1.0e-3,
            1.0,
        )
        g = jnp.asarray(
            [
                comp["lateral_force"] / (margin * self._config.lateral_force_limit) - 1.0,
                comp["axial_force"] / (margin * self._config.f_max) - 1.0,
                comp["bending_torque"] / (margin * self._config.bending_torque_limit) - 1.0,
                comp["torsional_torque"] / (margin * self._config.torsional_torque_limit) - 1.0,
            ]
        )
        return jnp.zeros((0,), dtype=g.dtype), g

    def _sequence_score_risk_with_margin(
        self,
        state,
        actions,
        aug_lambda=0.0,
        aug_rho=0.0,
        risk_safety_margin=1.0,
    ):
        margin = jnp.clip(
            jnp.asarray(risk_safety_margin, jnp.float32),
            1.0e-3,
            1.0,
        )

        def body(s, u):
            s2 = self.step(s, u)
            h, g = self.soft_feasibility_residual(s2, u)
            residual = jnp.concatenate([jnp.abs(h), jax.nn.relu(g)])
            penalty = aug_lambda * jnp.sum(residual) + 0.5 * aug_rho * jnp.sum(
                residual * residual
            )
            comp = self.wrench_components(s2.info["true_wrench"])
            force_violation = (
                (
                    comp["lateral_force"]
                    > margin * self._config.lateral_force_limit
                )
                | (comp["axial_force"] > margin * self._config.f_max)
            ).astype(jnp.float32)
            torque_violation = (
                (
                    comp["bending_torque"]
                    > margin * self._config.bending_torque_limit
                )
                | (
                    comp["torsional_torque"]
                    > margin * self._config.torsional_torque_limit
                )
            ).astype(jnp.float32)
            risk = jnp.asarray(
                [
                    force_violation,
                    torque_violation,
                    s2.info["jammed"],
                    jnp.abs(comp["axial_force"] - self._config.f_target)
                    / max(self._config.f_max, 1.0),
                ]
            )
            return s2, (s2.reward - penalty, risk)

        _, (score, per_step_risk) = jax.lax.scan(body, state, actions)
        # Certify the same short horizon used by realized controllability (and
        # never shorter than the action-delay window).  Performance/force
        # tracking still use the complete rollout horizon.
        commit_horizon = self.risk_commit_horizon(
            state, int(actions.shape[0])
        )
        committed_mask = (
            jnp.arange(per_step_risk.shape[0]) < commit_horizon
        )[:, None]
        committed = jnp.where(
            committed_mask,
            per_step_risk[:, :3],
            jnp.zeros_like(per_step_risk[:, :3]),
        )
        risk = jnp.asarray(
            [
                jnp.max(committed[:, 0]),
                jnp.max(committed[:, 1]),
                jnp.max(committed[:, 2]),
                jnp.mean(per_step_risk[:, 3]),
            ]
        )
        return jnp.mean(score), risk

    def sequence_score_risk(self, state, actions, aug_lambda=0.0, aug_rho=0.0):
        """Scores performance candidates against the robust inner safe set."""
        return self._sequence_score_risk_with_margin(
            state,
            actions,
            aug_lambda,
            aug_rho,
            self._config.risk_safety_margin,
        )

    def emergency_sequence_score_risk(
        self, state, actions, aug_lambda=0.0, aug_rho=0.0
    ):
        """Certifies the task-owned unload plan against physical limits.

        The robust inner set is deliberately used to reject ordinary
        performance candidates before deployment.  If the measured state is
        already inside that buffer band, however, no action can erase the
        current contact wrench instantaneously.  The zero-force/compliant
        emergency therefore receives a final check against the original
        physical safe set; reported violations use that same set.
        """
        return self._sequence_score_risk_with_margin(
            state, actions, aug_lambda, aug_rho, 1.0
        )

    def risk_commit_horizon(self, state, rollout_horizon: int):
        """Task-phase horizon for hard risk revalidation.

        Far above the socket, only the next submitted action (and any delayed
        actions) is committed before a new measurement.  Once the peg tip has
        crossed the socket plane, contact can begin during the next control
        interval even when MuJoCo still reports zero active contacts.  Enter
        the longer controllability-probe horizon at that geometric boundary,
        and retain it throughout measured contact.
        """
        action_delay = state.info["execution_action_delay_steps"]
        horizon = jnp.asarray(rollout_horizon, jnp.int32)
        no_contact_horizon = jnp.minimum(
            jnp.maximum(action_delay + 1, 1), horizon
        )
        # Replanning can react after the first observed stall when execution is
        # immediate.  With a queued action, however, one additional command is
        # already committed before that observation can change the input.  The
        # hard-risk horizon must therefore cover the remaining jam dwell plus
        # the real execution delay; otherwise a 3-step certificate can never
        # observe this task's 4-step jam event under one-step latency.
        observed_stall_credit = jnp.maximum(
            state.info["stall_steps"], jnp.asarray(1, jnp.int32)
        )
        remaining_jam_dwell = jnp.maximum(
            jnp.asarray(self._config.jam_dwell_steps, jnp.int32)
            - observed_stall_credit,
            jnp.asarray(1, jnp.int32),
        )
        latency_jam_horizon = jnp.where(
            action_delay > 0,
            action_delay + remaining_jam_dwell,
            action_delay + 1,
        )
        contact_horizon = jnp.minimum(
            jnp.maximum(
                jnp.maximum(
                    jnp.asarray(self._config.realization_probe_horizon, jnp.int32),
                    jnp.maximum(action_delay + 1, latency_jam_horizon),
                ),
                1,
            ),
            horizon,
        )
        actual_pose, _ = self._actual_pose(state.pipeline_state)
        commit_distance = jnp.abs(self._translation_step[2]) * contact_horizon
        near_or_in_contact = (
            (actual_pose[2] >= -commit_distance)
            | (state.info["prev_depth"] > 0.0)
            | (state.info["contact_count"] > 0)
        )
        return jnp.where(
            near_or_in_contact, contact_horizon, no_contact_horizon
        )

    def sequence_risk(self, state, actions):
        return self.sequence_score_risk(state, actions)[1]

    def sequence_risk_is_safe(self, risk):
        """Task-owned hard-safety interpretation of the sequence risk vector.

        Force-target error is a performance quantity.  Force/torque violations
        and jamming are the three hard contact-safety events for insertion.
        Keeping this interpretation in the task prevents the generic MGA
        backend from assuming that every environment uses the same risk schema.
        """
        return jnp.all(risk[:3] <= 1.0e-8)

    def _capture_lateral_radius(self, pose_z):
        """Conservative free aperture at the peg tip's current depth.

        The chamfer is widest at the socket plane and narrows to the bore
        clearance over ``chamfer_depth``.  OOD execution must use the known
        support lower bound rather than either the nominal clearance or the
        hidden randomized draw; this keeps the safety contract robust without
        leaking execution parameters into planning observations.
        """
        cfg = self._config
        clearance = (
            min(float(cfg.clearance), float(cfg.ood_clearance_range[0]))
            if str(cfg.level).lower() == "ood"
            else float(cfg.clearance)
        )
        chamfer_fraction = jnp.clip(
            1.0 - jnp.maximum(pose_z, 0.0) / max(cfg.chamfer_depth, 1.0e-6),
            0.0,
            1.0,
        )
        return clearance + cfg.chamfer_width * chamfer_fraction

    def _capture_is_stable(self, state, pose, angle):
        """Current-pose certificate used to enter/leave task recovery."""
        del state
        return (
            jnp.linalg.norm(pose[:2])
            <= self._capture_lateral_radius(pose[2])
        ) & (jnp.linalg.norm(angle) <= self._config.success_angle_tol)

    def project_mga_candidate(self, state, nodes):
        """Projects a geometry-updated MGA plan into the executable set.

        Geometry/retraction can move node values outside the sampler's action
        box.  The physical environment clips such commands only at execution,
        which otherwise makes rollout/revalidation certify a different plan.
        Peg insertion additionally constrains the committed (first) node after
        a contact-count transition once the next command can enter the deeply
        constrained half of the bore.  With delayed action/measurement queues,
        positive axial motion is held until the stale-contact memory expires;
        without delay, the immediate re-entry step is merely slowed.  Future
        nodes remain free because receding-horizon replanning revalidates them
        before submission.  Near the socket, the first node is also projected
        through the already committed action-delay queue.  Only its lateral
        command is tightened to the robust aperture; axial progress and later
        nodes remain available to the optimizer.
        """
        projected = jnp.clip(nodes, -1.0, 1.0)
        volatile = (
            state.info["contact_volatility"]
            > self._config.contact_switch_volatility_threshold
        )
        unstable_contact_set = (
            (state.info["contact_count"] <= 0.0)
            | (jnp.abs(state.info["contact_count_delta"]) > 0.0)
        )
        total_delay = (
            state.info["execution_action_delay_steps"]
            + state.info["execution_sensor_delay_steps"]
        )
        delayed_volatility_threshold = (
            self._config.contact_switch_volatility_threshold
            * jnp.power(
                jnp.asarray(0.5, projected.dtype),
                total_delay.astype(projected.dtype),
            )
        )
        delayed_contact_memory = (
            (total_delay > 0)
            & (
                state.info["contact_volatility"]
                > delayed_volatility_threshold
            )
        )
        pose, angle = self._actual_pose(state.pipeline_state)
        outside_capture = ~self._capture_is_stable(state, pose, angle)
        contact_reach = (
            jnp.abs(self._translation_step[2])
            * float(max(int(self._config.realization_probe_horizon), 1))
        )
        unloaded_capture_recovery = (
            (state.info["contact_count"] <= 0)
            & (pose[2] >= -contact_reach)
            & outside_capture
        )
        reaches_deep_bore = (
            state.info["prev_depth"] + jnp.abs(self._translation_step[2])
            >= 0.5 * self._config.socket_depth
        )
        recent_contact_switch = (
            reaches_deep_bore
            & ((volatile & unstable_contact_set) | delayed_contact_memory)
        )
        immediate_switch_limit = jnp.clip(
            jnp.asarray(
                self._config.contact_switch_axial_action_limit,
                projected.dtype,
            ),
            0.0,
            1.0,
        )
        switch_limit = jnp.where(
            delayed_contact_memory,
            jnp.asarray(0.0, projected.dtype),
            immediate_switch_limit,
        )
        delayed_precontact = (
            (total_delay > 0)
            & (state.info["contact_count"] <= 0)
            & (pose[2] >= -contact_reach)
        )
        # A delayed contact-set transition means node zero will execute before
        # the newly observed wrench can affect another decision.  Freeze only
        # that committed pose increment; future nodes remain available for
        # receding-horizon recovery.  Axial motion is handled by the same
        # delayed switch limit below.
        # Freeze lateral/orientation motion while contact memory is stale, but
        # preserve axial sign.  The bound below independently clips positive
        # axial motion to zero and must still allow a task-owned negative
        # retract; zeroing all six pose channels here previously erased that
        # safety action until the delay memory expired.
        frozen_committed_pose = (
            projected[0, :6].at[:2].set(0.0).at[3:6].set(0.0)
        )
        committed_pose = jnp.where(
            delayed_contact_memory & (~unloaded_capture_recovery),
            frozen_committed_pose,
            projected[0, :6],
        )
        projected = projected.at[0, :6].set(committed_pose)
        axial_upper = jnp.where(
            recent_contact_switch,
            switch_limit,
            jnp.asarray(1.0, projected.dtype),
        )
        axial_upper = jnp.minimum(
            axial_upper,
            jnp.where(
                delayed_precontact,
                # A queued command cannot be cancelled at the next replan.
                # Halve the immediate contact-switch bandwidth so committed
                # axial motion stays inside the validated braking envelope.
                0.5 * immediate_switch_limit,
                jnp.asarray(1.0, projected.dtype),
            ),
        )
        projected = projected.at[0, 2].set(
            jnp.clip(projected[0, 2], -1.0, axial_upper)
        )

        # Certify the command that node zero will produce after every action
        # already committed to the execution-delay queue.  Bounding only the
        # freshly sampled delta is insufficient: commands accumulate in
        # ``command_pose`` and a individually small lateral action can still
        # place the impedance target beyond the bore.  Reserve the current
        # command-to-tip tracking error with the triangle inequality so the
        # target does not consume clearance already used by servo lag.
        delay = state.info["execution_action_delay_steps"]
        queue_mask = (
            jnp.arange(_QUEUE_CAPACITY, dtype=delay.dtype) < delay
        ).astype(projected.dtype)
        pose_scale = jnp.concatenate([
            self._translation_step,
            jnp.full((3,), self._config.rotation_step, projected.dtype),
        ])
        queued_pose_delta = jnp.sum(
            state.info["action_queue"][:, self.spec.r_slice]
            * queue_mask[:, None]
            * pose_scale[None, :],
            axis=0,
        )
        pre_candidate_command = state.info["command_pose"] + queued_pose_delta
        candidate_delta = projected[0, self.spec.r_slice] * pose_scale
        candidate_command = pre_candidate_command + candidate_delta
        command_near_socket = candidate_command[2] >= -contact_reach
        aperture = self._capture_lateral_radius(candidate_command[2])
        tracking_error = jnp.linalg.norm(
            pose[:2] - state.info["command_pose"][:2]
        )
        command_radius = jnp.maximum(aperture - tracking_error, 0.0)
        lateral_norm = jnp.linalg.norm(candidate_command[:2])
        lateral_scale = jnp.minimum(
            1.0,
            command_radius / jnp.maximum(lateral_norm, 1.0e-8),
        )
        certified_target = candidate_command[:2] * lateral_scale
        certified_raw = (
            certified_target - pre_candidate_command[:2]
        ) / self._translation_step[:2]
        aperture_projected_lateral = jnp.where(
            command_near_socket & (~unloaded_capture_recovery),
            jnp.clip(certified_raw, -1.0, 1.0),
            projected[0, :2],
        )
        projected = projected.at[0, :2].set(aperture_projected_lateral)
        return projected

    def emergency_plan_should_override(self, state):
        """Requires task recovery when physical stall/jam evidence is authoritative.

        Model-based candidate rollouts may use nominal contact geometry.  A jam
        precursor or jam already observed by the execution task is not a model
        prediction and must not be overwritten by a nominally safe refined
        plan.  Acting on the first sustained-stall sample leaves the delayed
        actuator one command interval to unload before the task's dwell counter
        declares a jam.  After the jam unloads, retain recentering only until
        the peg is back inside the chamfer capture region; releasing there
        avoids deadlocking the recovery before insertion progress can mark it
        as recovered.
        """
        pose, angle = self._actual_pose(state.pipeline_state)
        captured = self._capture_is_stable(state, pose, angle)
        unloaded = state.info["contact_count"] <= 0
        recovery_pending = (
            (state.info["jammed_once"] > 0.5)
            & (state.info["recovered"] < 0.5)
        )
        return (
            (state.info["jammed"] > 0.5)
            | (state.info["stall_steps"] > 0)
            | (recovery_pending & (~(captured & unloaded)))
        )

    def emergency_plan(self, state, reference_nodes, t0=0.0):
        """Returns a same-shape, executable unload-and-recenter horizon.

        This is deliberately a task action, not a solver heuristic.  It first
        retracts along socket-z with zero commanded force and reduced axial
        stiffness, then uses nominal lateral impedance to remove the measured
        lateral/orientation error while unloaded.  A delayed recovery suffix
        is outside the immediately committed safety prefix; after shifting it
        must be revalidated from the next measurement before it can execute.
        """
        del t0
        retract_action = jnp.zeros(
            (self.action_size,), dtype=reference_nodes.dtype
        )
        retract_action = retract_action.at[self.spec.s_slice].set(
            jnp.asarray(
                [0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                reference_nodes.dtype,
            )
        )
        retract_action = retract_action.at[self.spec.nu_slice].set(
            self._force_action_raw(self._config.f_min)
        )
        retract = jnp.broadcast_to(retract_action, reference_nodes.shape)
        node_count = int(reference_nodes.shape[0])
        retract_nodes = max(1, node_count // 2)
        # Zero force plus axial compliance unloads a currently safe contact
        # without erasing insertion progress.  A geometric retreat is needed
        # only after the *measured* wrench has already left the physical set or
        # the task declares a jam.  The old unconditional quarter-step was
        # repeatedly re-issued at low force limits and converted a safe hold
        # into systematic backout.
        measured = self.wrench_components(state.info["measured_wrench"])
        retract_required = (
            (measured["lateral_force"] > self._config.lateral_force_limit)
            | (measured["axial_force"] > self._config.f_max)
            | (measured["bending_torque"] > self._config.bending_torque_limit)
            | (measured["torsional_torque"] > self._config.torsional_torque_limit)
            | (state.info["stall_steps"] > 0)
            | (state.info["jammed"] > 0.5)
        )
        retract_step = jnp.where(retract_required, -0.25, 0.0)
        retract = retract.at[:retract_nodes, 2].set(retract_step)
        pose, angle = self._actual_pose(state.pipeline_state)
        # Spread the recovery over the interior control nodes.  Clipping keeps
        # every node executable; replanning then closes the loop if one horizon
        # cannot remove the complete offset.
        recovery_nodes = max(node_count - 2, 1)
        translation_correction_raw = jnp.clip(
            -pose[:2]
            / (self._translation_step[:2] * float(recovery_nodes)),
            -1.0,
            1.0,
        )
        translation_correction = jnp.where(
            jnp.linalg.norm(pose[:2])
            > self._capture_lateral_radius(pose[2]),
            translation_correction_raw,
            jnp.zeros_like(translation_correction_raw),
        )
        # ``_actual_pose`` uses orientation_error(actual, socket), whose
        # small-angle sign is socket-minus-actual.  The desired command must
        # therefore move in the SAME sign as this value.  Using ``-angle``
        # drives the peg farther into the measured misalignment and can wedge
        # it when a recovery node is actually deployed.
        rotation_correction_raw = jnp.clip(
            angle / (self._config.rotation_step * float(recovery_nodes)),
            -1.0,
            1.0,
        )
        rotation_correction = jnp.where(
            jnp.linalg.norm(angle) > self._config.success_angle_tol,
            rotation_correction_raw,
            jnp.zeros_like(rotation_correction_raw),
        )
        if node_count > 2:
            retract = retract.at[1:-1, :2].set(translation_correction)
            retract = retract.at[1:-1, 3:6].set(rotation_correction)
        # MPC executes node zero before it is allowed to replan.  Begin the
        # recenter motion there with the smallest horizon-derived share while
        # the peg may still be loaded; use available wrench slack to recover
        # the complete set violation within the finite episode.  The
        # set-valued deadband above makes this different from an unconstrained
        # second integrator: once lateral/orientation state re-enters the
        # admissible capture set, the corresponding increment becomes zero.
        base_share = 1.0 / float(max(node_count, 1))
        measured_lateral_ratio = measured["lateral_force"] / max(
            self._config.lateral_force_limit, 1.0e-6
        )
        measured_torque_ratio = measured["bending_torque"] / max(
            self._config.bending_torque_limit, 1.0e-6
        )
        wrench_slack = jnp.clip(
            1.0 - jnp.maximum(measured_lateral_ratio, measured_torque_ratio),
            0.0,
            1.0,
        )
        # Zero measured wrench is informative in distant free space, but not
        # after the peg has entered the one-probe-horizon capture region: rim
        # contact can occur before the next measurement.  Do not turn that
        # unobserved-contact interval into a full-speed committed correction.
        contact_reach = (
            jnp.abs(self._translation_step[2])
            * float(max(int(self._config.realization_probe_horizon), 1))
        )
        contact_imminent = (
            (pose[2] >= -contact_reach)
            | (state.info["prev_depth"] > 0.0)
        )
        outside_capture = ~self._capture_is_stable(state, pose, angle)
        unloaded_capture_recovery = (
            contact_imminent
            & (state.info["contact_count"] <= 0)
            & outside_capture
        )
        unobserved_contact = (
            contact_imminent
            & (state.info["contact_count"] <= 0)
            & (~unloaded_capture_recovery)
        )
        committed_motion_slack = jnp.where(
            unobserved_contact, 0.0, wrench_slack
        )
        first_node_share = base_share + (
            float(recovery_nodes) - base_share
        ) * committed_motion_slack
        retract = retract.at[0, :2].set(
            jnp.clip(first_node_share * translation_correction, -1.0, 1.0)
        )
        retract = retract.at[0, 3:6].set(
            jnp.clip(first_node_share * rotation_correction, -1.0, 1.0)
        )
        if node_count > 2:
            retract = retract.at[1, :2].set(translation_correction)
            retract = retract.at[1, 3:6].set(rotation_correction)
        # The terminal node is intentionally pose-neutral; terminal-hold shift
        # therefore preserves zero force/compliance without accumulating pose.
        retract = retract.at[-1, :6].set(0.0)
        # Keep the complete committed prefix at zero force.  The later nodes
        # form a task-owned, axially compliant recovery incumbent: modest
        # insertion motion at the nominal target force, followed by a
        # pose-neutral force hold.  Receding-horizon acceptance evaluates this
        # suffix again after the zero-force prefix is executed, so it is not an
        # unverified action escape from the emergency contract.
        recovery_start = min(max(retract_nodes, 2), node_count)
        if recovery_start < node_count:
            retract = retract.at[
                recovery_start:, self.spec.nu_slice
            ].set(self._force_action_raw(self._config.f_target))
        if recovery_start < node_count - 1:
            retract = retract.at[recovery_start:-1, 2].set(0.5)
        # While physically loaded, every node in the committed unload prefix
        # must remain lateral/orientation neutral.  Apply this mask last so no
        # suffix construction or indexed update can reintroduce recovery motion
        # before the next measurement and revalidation.
        loaded_prefix = (
            (state.info["contact_count"] > 0)
            & (jnp.arange(node_count) < retract_nodes)
        )[:, None]
        neutral_lateral = jnp.where(
            loaded_prefix, jnp.zeros_like(retract[:, :2]), retract[:, :2]
        )
        neutral_rotation = jnp.where(
            loaded_prefix, jnp.zeros_like(retract[:, 3:6]), retract[:, 3:6]
        )
        retract = jnp.concatenate([
            neutral_lateral,
            retract[:, 2:3],
            neutral_rotation,
            retract[:, 6:],
        ], axis=-1)
        return retract

    def emergency_plan_is_active(self, nodes):
        """Identifies a pure zero-force/compliant safety hold.

        Plans with a delayed recovery suffix intentionally return false after
        shift so that suffix is treated as a performance incumbent and fully
        revalidated before deployment.
        """
        expected_force = self._force_action_raw(self._config.f_min)
        force_match = jnp.all(
            jnp.abs(nodes[:, self.spec.nu_slice] - expected_force) < 1.0e-3
        )
        raw_s = nodes[:, self.spec.s_slice]
        axial_compliant = jnp.all(raw_s[:, 5] < -0.8)
        return force_match & axial_compliant

    def safety_index(self, state):
        comp = self.wrench_components(state.info["true_wrench"])
        return jnp.max(
            jnp.asarray(
                [
                    comp["lateral_force"] / self._config.lateral_force_limit - 1.0,
                    comp["axial_force"] / self._config.f_max - 1.0,
                    comp["bending_torque"] / self._config.bending_torque_limit - 1.0,
                    comp["torsional_torque"] / self._config.torsional_torque_limit - 1.0,
                ]
            )
        )


class PegInsertDomainEnv:
    """Reset-key randomized family of shape-compatible insertion domains."""

    def __init__(self, domains):
        self.domains = tuple(domains)
        if not self.domains:
            raise ValueError("PegInsertDomainEnv needs at least one domain")
        action_sizes = {int(env.action_size) for env in self.domains}
        observation_sizes = {int(env.observation_size) for env in self.domains}
        if len(action_sizes) != 1 or len(observation_sizes) != 1:
            raise ValueError(
                "all randomized peg-insert domains must share action/observation sizes"
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
        return state.replace(info={**state.info, "_rl_domain_index": domain_index})

    def step(self, state, action):
        domain_index = state.info["_rl_domain_index"]
        branches = tuple(
            (lambda s, env=env: env.step(s, action)) for env in self.domains
        )
        return jax.lax.switch(domain_index, branches, state)


class PegInsertResidualActionEnv:
    """Train a residual policy around an executable insertion primitive."""

    def __init__(self, env, action_bias, action_scale=None):
        self.env = env
        self.action_bias = jnp.asarray(action_bias, dtype=jnp.float32)
        expected = (int(env.action_size),)
        if self.action_bias.shape != expected:
            raise ValueError(
                f"action_bias must match the insertion primitive: "
                f"{self.action_bias.shape} != {expected}"
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


def make_execution_pair(
    model_env: PegInsertEnv,
    *,
    level: str = "ood",
    domain_seed: int = 0,
    **overrides,
):
    """Create a topology-compatible hidden execution model for P3 evaluation."""
    cfg = replace(
        model_env._config,
        level=level,
        domain_seed=int(domain_seed),
        **overrides,
    )
    execution = PegInsertEnv(cfg)
    if execution.action_size != model_env.action_size:
        raise ValueError("nominal and execution insertion action spaces differ")
    if execution.sys.q_size() != model_env.sys.q_size() or execution.sys.qd_size() != model_env.sys.qd_size():
        raise ValueError("nominal and execution insertion state topology differs")
    return execution


__all__ = [
    "PegInsertConfig",
    "PegInsertEnv",
    "PegInsertDomainEnv",
    "PegInsertResidualActionEnv",
    "make_execution_pair",
]
