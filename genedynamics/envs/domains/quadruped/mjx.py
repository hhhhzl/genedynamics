"""
Quadruped MJX (MuJoCo XLA) environment for JAX-based planning.

Provides JAX-compatible transition for MBD, 2GO, and other diffusion solvers.
Uses mujoco-mjx for GPU/TPU-accelerated, differentiable physics simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import mujoco
    from mujoco import mjx
    import jax
    import jax.numpy as jnp
    MJX_AVAILABLE = True
except ImportError:
    MJX_AVAILABLE = False
    mujoco = None
    mjx = None
    jax = None
    jnp = None


def _resolve_quadruped_model_path(model: str) -> Optional[str]:
    """Resolve MJCF path via robot registry. Prefer MJX-compatible XML when available."""
    if model == "go2":
        try:
            from genedynamics.robots.registry import _get_go2_path
            path = _get_go2_path(mjx=True)
            if path:
                return path
        except Exception:
            pass
        # Fallback: project third_party/mujoco_menagerie
        # parents[4] == project root (this file is envs/domains/quadruped/mjx.py)
        proj = Path(__file__).resolve().parents[4]
        for f in ("go2_mjx.xml", "go2.xml"):
            p = proj / "third_party" / "mujoco_menagerie" / "unitree_go2" / f
            if p.exists():
                return str(p)
    try:
        from genedynamics.robots import get_robot_registry
        reg = get_robot_registry()
        path = reg.get_model_path("quadruped", model)
        if path:
            return path
    except Exception:
        pass
    if model in ("ant", "flat", "rough", "push"):
        try:
            from genedynamics.robots.registry import _get_ant_path
            return _get_ant_path()
        except Exception:
            pass
    if model == "go2":
        try:
            from genedynamics.robots.registry import _get_go2_path
            return _get_go2_path(mjx=True) or None
        except Exception:
            pass
    return None


@dataclass
class QuadrupedMjxEnv:
    """
    Quadruped environment with MJX (MuJoCo XLA) physics backend.

    Provides JAX-compatible jax_transition/jax_model_transition for diffusion
    solvers (MBD, 2GO, etc.). State format: flat [qpos; qvel].
    Action: joint torques, clipped to [-control_limit, control_limit].

    Attributes:
        dt: Simulation timestep (s)
        horizon: Planning horizon (steps)
        act_dim: Action dimension (8 for ant, 12 for Go2)
        control_limit: Torque limit (symmetric)
        target: Goal position (x, y, z)
        model: Robot model id ("ant", "go2", etc.)
        model_path: Override path to MJCF (optional)
        obstacles: Optional ObstacleManager for jax_sdf (Phase 5)
    """

    dt: float = 0.05
    horizon: int = 200
    act_dim: int = 8
    control_limit: float = 1.0
    target: Tuple[float, float, float] = (2.0, 0.0, 0.5)
    desired_velocity: Optional[Tuple[float, float, float]] = None
    model: str = "ant"
    model_path: Optional[str] = None
    obstacles: Optional[Any] = None

    _model: Any = field(default=None, repr=False)
    _mjx_model: Any = field(default=None, repr=False)
    _home_qpos: Optional[np.ndarray] = field(default=None, repr=False)
    _home_ctrl: Optional[np.ndarray] = field(default=None, repr=False)
    _ctrl_min: Optional[np.ndarray] = field(default=None, repr=False)
    _ctrl_max: Optional[np.ndarray] = field(default=None, repr=False)
    _base_z_min: float = field(default=0.18, repr=False)
    _max_state_abs: float = field(default=800.0, repr=False)
    _max_qvel_abs: float = field(default=120.0, repr=False)
    _max_pos_jump: float = field(default=0.8, repr=False)
    _max_ang_vel_abs: float = field(default=60.0, repr=False)
    _min_upright_cos: float = field(default=-1.0, repr=False)
    nq: int = 15
    nv: int = 14
    state_dim: int = 29
    nu: int = 8

    def __post_init__(self) -> None:
        if not MJX_AVAILABLE:
            raise ImportError(
                "QuadrupedMjxEnv requires mujoco, mujoco-mjx, and jax. "
                "Install with: pip install mujoco mujoco-mjx jax jaxlib"
            )
        path = self.model_path or _resolve_quadruped_model_path(self.model)
        if not path or not Path(path).exists():
            raise FileNotFoundError(
                f"Quadruped model not found for '{self.model}'. "
                f"Path: {path}. Install gymnasium for ant, or set MUJOCO_MENAGERIE_PATH for Go2."
            )
        self._load_model(path)

    def _load_model(self, model_path: str) -> None:
        """Load MuJoCo model and create MJX device arrays."""
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._model.opt.timestep = self.dt
        # MJX only supports mjSOL_CG and mjSOL_NEWTON (not mjSOL_PGS)
        if self._model.opt.solver == mujoco.mjtSolver.mjSOL_PGS:
            self._model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        self._mjx_model = mjx.put_model(self._model)
        # A3: Explicitly place MJX model on GPU when backend uses cuda (avoids default-device ambiguity)
        try:
            from genedynamics.core.backends.runtime import RuntimeBackendManager
            backend = RuntimeBackendManager.get_backend()
            if backend and getattr(backend, "device", None) in ("cuda", "gpu"):
                gpu_devices = jax.devices("gpu")
                if gpu_devices:
                    self._mjx_model = jax.device_put(self._mjx_model, gpu_devices[0])
        except Exception:
            pass
        self.nq = self._model.nq
        self.nv = self._model.nv
        self.nu = self._model.nu
        self.act_dim = self.nu
        self.state_dim = self.nq + self.nv
        self._home_qpos = np.asarray(self._model.qpos0, dtype=np.float32).copy()
        self._home_ctrl = np.zeros(self.nu, dtype=np.float32)
        if self._model.actuator_ctrlrange.shape[0] == self.nu:
            ctrlrange = np.asarray(self._model.actuator_ctrlrange, dtype=np.float32)
            self._ctrl_min = ctrlrange[:, 0].copy()
            self._ctrl_max = ctrlrange[:, 1].copy()
        else:
            lim = float(self.control_limit)
            self._ctrl_min = np.full(self.nu, -lim, dtype=np.float32)
            self._ctrl_max = np.full(self.nu, lim, dtype=np.float32)

        # Prefer explicit "home" keyframe for stable legged initialization/control.
        home_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_id < 0 and self._model.nkey > 0:
            home_id = 0
        if home_id >= 0:
            key_qpos = np.asarray(self._model.key_qpos, dtype=np.float32).reshape(self._model.nkey, self.nq)
            self._home_qpos = key_qpos[home_id].copy()
            if self._model.key_ctrl.size >= self._model.nkey * self.nu:
                key_ctrl = np.asarray(self._model.key_ctrl, dtype=np.float32).reshape(self._model.nkey, self.nu)
                self._home_ctrl = key_ctrl[home_id].copy()
        # Hard floor projection threshold for floating base z.
        home_z = float(self._home_qpos[2]) if self._home_qpos is not None and self._home_qpos.size >= 3 else 0.27
        # Keep floating base near standing height; too-low z quickly destabilizes legged plans.
        self._base_z_min = max(0.05, home_z - 0.05)
        if str(self.model).lower() == "go2":
            # Go2 is sensitive to velocity spikes; use tighter safety thresholds.
            self._max_state_abs = 500.0
            self._max_qvel_abs = 60.0
            self._max_pos_jump = 0.45
            self._max_ang_vel_abs = 25.0
            self._min_upright_cos = 0.20

    def reset(self, rng: Optional[Any] = None, seed: Optional[int] = None, **kwargs: Any) -> Tuple[np.ndarray, Dict]:
        """Reset to model home pose with zero velocity."""
        if seed is not None:
            np.random.seed(seed)
        qpos = np.asarray(self._home_qpos, dtype=np.float32).copy()
        qvel = np.zeros(self.nv, dtype=np.float32)
        if qpos.size >= 3:
            qpos[2] = max(float(qpos[2]), float(self._base_z_min))
        if qpos.size >= 7:
            quat_norm = float(np.linalg.norm(qpos[3:7]))
            if not np.isfinite(quat_norm) or quat_norm < 1e-6:
                qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            else:
                qpos[3:7] = qpos[3:7] / quat_norm
        flat = np.concatenate([qpos, qvel]).astype(np.float32)
        return flat, {}

    def transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """NumPy transition for execution. state/action -> next_state."""
        state = np.asarray(state, dtype=np.float32).ravel()
        action = np.asarray(action, dtype=np.float32).ravel()
        action = np.clip(action, -self.control_limit, self.control_limit)
        if action.size > self.nu:
            action = action[: self.nu]
        elif action.size < self.nu:
            action = np.pad(action, (0, self.nu - action.size), constant_values=0.0)
        ctrl_base = np.asarray(self._home_ctrl, dtype=np.float32).ravel()
        ctrl = np.clip(ctrl_base + action, self._ctrl_min, self._ctrl_max)

        data = mjx.make_data(self._mjx_model)
        data = data.replace(
            qpos=jnp.asarray(state[: self.nq], dtype=jnp.float32),
            qvel=jnp.asarray(state[self.nq : self.nq + self.nv], dtype=jnp.float32),
        )
        data = data.replace(ctrl=jnp.asarray(ctrl, dtype=jnp.float32))
        next_data = mjx.step(self._mjx_model, data)
        next_qpos = np.asarray(next_data.qpos, dtype=np.float32).copy()
        next_qvel = np.asarray(next_data.qvel, dtype=np.float32).copy()
        if next_qvel.size > 0:
            next_qvel = np.clip(next_qvel, -self._max_qvel_abs, self._max_qvel_abs)
        if next_qpos.size >= 3 and next_qpos[2] < self._base_z_min:
            next_qpos[2] = self._base_z_min
            if next_qvel.size >= 3:
                next_qvel[2] = max(0.0, float(next_qvel[2]))
        if next_qpos.size >= 7:
            quat = next_qpos[3:7]
            qn = float(np.linalg.norm(quat))
            if np.isfinite(qn) and qn > 1e-6:
                next_qpos[3:7] = quat / qn
        next_state = np.concatenate([next_qpos, next_qvel])
        prev_pos = np.asarray(state[:3], dtype=np.float32) if state.size >= 3 else np.zeros(3, dtype=np.float32)
        next_pos = np.asarray(next_qpos[:3], dtype=np.float32) if next_qpos.size >= 3 else np.zeros(3, dtype=np.float32)
        pos_jump = float(np.linalg.norm(next_pos - prev_pos))
        z_ok = bool(self._base_z_min <= float(next_pos[2]) <= 1.5) if next_pos.size >= 3 else True
        vel_ok = bool(np.max(np.abs(next_qvel)) <= self._max_qvel_abs) if next_qvel.size else True
        ang_vel_ok = bool(np.max(np.abs(next_qvel[3:6])) <= self._max_ang_vel_abs) if next_qvel.size >= 6 else True
        if next_qpos.size >= 7:
            qx, qy = float(next_qpos[4]), float(next_qpos[5])
            upright_cos = 1.0 - 2.0 * (qx * qx + qy * qy)
            upright_ok = upright_cos >= self._min_upright_cos
        else:
            upright_ok = True
        if (
            (not np.all(np.isfinite(next_state)))
            or np.max(np.abs(next_state)) > self._max_state_abs
            or (not z_ok)
            or (not vel_ok)
            or (not ang_vel_ok)
            or (not upright_ok)
            or pos_jump > self._max_pos_jump
        ):
            # Guard against occasional MJX blow-ups: keep configuration, zero velocity.
            next_state = np.concatenate([state[: self.nq], np.zeros(self.nv, dtype=np.float32)])
        return next_state

    def jax_transition(self, state: Any, action: Any) -> Any:
        """
        JAX-compatible transition for planning. Pure function, supports vmap.

        Args:
            state: (nq+nv,) or (B, nq+nv) JAX array
            action: (nu,) or (B, nu) JAX array

        Returns:
            next_state: same shape as state
        """
        state = jnp.asarray(state, dtype=jnp.float32)
        action = jnp.asarray(action, dtype=jnp.float32)
        action = jnp.clip(action, -self.control_limit, self.control_limit)
        ctrl_base = jnp.asarray(self._home_ctrl, dtype=action.dtype)
        ctrl_min = jnp.asarray(self._ctrl_min, dtype=action.dtype)
        ctrl_max = jnp.asarray(self._ctrl_max, dtype=action.dtype)

        def _step(s: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
            a_flat = jnp.reshape(a, (-1))
            a_pad = jnp.concatenate([a_flat, jnp.zeros(self.nu, dtype=a_flat.dtype)])[: self.nu]
            ctrl = self._compute_ctrl(
                s[: self.nq], s[self.nq : self.nq + self.nv], a_pad,
                ctrl_base, ctrl_min, ctrl_max,
            )
            data = mjx.make_data(self._mjx_model)
            data = data.replace(
                qpos=s[: self.nq],
                qvel=s[self.nq : self.nq + self.nv],
            )
            data = data.replace(ctrl=ctrl)
            next_data = mjx.step(self._mjx_model, data)
            next_qpos = next_data.qpos
            if self.nq >= 3:
                z_clamped = jnp.maximum(next_qpos[2], jnp.asarray(self._base_z_min, dtype=next_qpos.dtype))
                next_qpos = next_qpos.at[2].set(z_clamped)
            if self.nq >= 7:
                quat = next_qpos[3:7]
                qn = jnp.linalg.norm(quat)
                quat_norm = jnp.where(qn > 1e-6, quat / qn, jnp.array([1.0, 0.0, 0.0, 0.0], dtype=quat.dtype))
                next_qpos = next_qpos.at[3:7].set(quat_norm)
            next_qvel = next_data.qvel
            next_qvel = jnp.clip(next_qvel, -self._max_qvel_abs, self._max_qvel_abs)
            if self.nv >= 3 and self.nq >= 3:
                next_qvel = next_qvel.at[2].set(
                    jnp.where(next_qpos[2] <= self._base_z_min + 1e-6, jnp.maximum(next_qvel[2], 0.0), next_qvel[2])
                )
            next_state = jnp.concatenate([next_qpos, next_qvel])
            next_pos = next_qpos[:3] if self.nq >= 3 else jnp.zeros((3,), dtype=s.dtype)
            prev_pos = s[:3] if self.nq >= 3 else jnp.zeros((3,), dtype=s.dtype)
            pos_jump = jnp.linalg.norm(next_pos - prev_pos)
            z_ok = jnp.logical_and(next_pos[2] >= self._base_z_min, next_pos[2] <= 1.5) if self.nq >= 3 else True
            ang_vel_ok = jnp.max(jnp.abs(next_qvel[3:6])) <= self._max_ang_vel_abs if self.nv >= 6 else True
            if self.nq >= 7:
                qx = next_qpos[4]
                qy = next_qpos[5]
                upright_cos = 1.0 - 2.0 * (qx * qx + qy * qy)
                upright_ok = upright_cos >= self._min_upright_cos
            else:
                upright_ok = True
            sane = jnp.logical_and(
                jnp.logical_and(jnp.all(jnp.isfinite(next_state)), jnp.max(jnp.abs(next_state)) <= self._max_state_abs),
                jnp.logical_and(
                    jnp.logical_and(jnp.max(jnp.abs(next_qvel)) <= self._max_qvel_abs, jnp.logical_and(z_ok, pos_jump <= self._max_pos_jump)),
                    jnp.logical_and(ang_vel_ok, upright_ok),
                ),
            )
            safe_state = jnp.concatenate([s[: self.nq], jnp.zeros(self.nv, dtype=s.dtype)])
            return jnp.where(sane, next_state, safe_state)

        if state.ndim == 1:
            return _step(state, action)
        return jax.vmap(_step)(state, action)

    def _compute_ctrl(self, qpos, qvel, a_pad, ctrl_base, ctrl_min, ctrl_max):
        """Map the (padded) action to actuator ctrl. Default: torque offset from
        the home ctrl. Subclasses (e.g. DIAL trot) override for act2tau PD."""
        return jnp.clip(ctrl_base + a_pad, ctrl_min, ctrl_max)

    def jax_model_transition(self, state: Any, action: Any) -> Any:
        """Same as jax_transition (no state projection for quadruped)."""
        return self.jax_transition(state, action)

    def jax_sdf(self, pos: Any) -> Any:
        """
        JAX SDF for obstacle avoidance (Phase 5).
        pos: (..., 3) positions. Returns min distance to obstacles.
        """
        if jnp is None:
            raise RuntimeError("JAX required for jax_sdf")
        if self.obstacles is None or len(self.obstacles) == 0:
            return jnp.ones(pos.shape[:-1], dtype=jnp.float32) * 1e5
        if hasattr(self.obstacles, "jax_sdf"):
            return self.obstacles.jax_sdf(pos)
        prims = [p for p in self.obstacles.obstacles if hasattr(p, "jax_sdf")]
        if not prims:
            return jnp.ones(pos.shape[:-1], dtype=jnp.float32) * 1e5
        sdfs = [p.jax_sdf(pos) for p in prims]
        if len(sdfs) == 1:
            return sdfs[0]
        return jnp.min(jnp.stack(sdfs, axis=0), axis=0)

    def cost(self, state: np.ndarray) -> float:
        """Task cost: base position error + velocity penalty."""
        pos = np.asarray(state[:3], dtype=np.float32)
        vel = np.asarray(
            state[self.nq : self.nq + 3] if state.size > self.nq + 3 else [0, 0, 0],
            dtype=np.float32,
        )
        target = np.asarray(self.target, dtype=np.float32)
        pos_err = float(np.sum((pos - target) ** 2))
        vel_err = 0.1 * float(np.sum(vel ** 2))
        return pos_err + vel_err

    def close(self) -> None:
        """Release resources."""
        self._model = None
        self._mjx_model = None


@dataclass
class QuadrupedAntMjxEnv(QuadrupedMjxEnv):
    """Ant quadruped with MJX (gymnasium ant.xml)."""

    model: str = "ant"


@dataclass
class QuadrupedGo2MjxEnv(QuadrupedMjxEnv):
    """Unitree Go2 quadruped with MJX (mujoco_menagerie).

    NOTE: ``@dataclass`` is required so the ``model="go2"`` override actually
    takes effect; without it the inherited dataclass __init__ would default
    ``model`` to the parent's "ant" and silently load the Ant model.
    """

    model: str = "go2"
    act_dim: int = 12
    control_limit: float = 1.0


# --- DIAL-MPC go2 trot (general integration: a normal registered env, no /dial dir) ---
from genedynamics.core.registry.environments import register_environment  # noqa: E402


def _trot_foot_step(duty_ratio, cadence, amplitude, phases, time):
    """Target swing-foot height per leg (port of dial-mpc get_foot_step)."""
    def step_height(t, footphase, dr):
        angle = (t + jnp.pi - footphase) % (2 * jnp.pi) - jnp.pi
        angle = jnp.where(dr < 1, angle * 0.5 / (1 - dr), angle)
        clipped = jnp.clip(angle, -jnp.pi / 2, jnp.pi / 2)
        value = jnp.where(dr < 1, jnp.cos(clipped), 0.0)
        return jnp.where(jnp.abs(value) >= 1e-6, jnp.abs(value), 0.0)
    return amplitude * jax.vmap(step_height, in_axes=(None, 0, None))(
        time * 2 * jnp.pi * cadence + jnp.pi, 2 * jnp.pi * phases, duty_ratio
    )


@register_environment("quadruped_go2_mjx_trot", "quadruped_go2_dial")
@dataclass
class QuadrupedGo2TrotMjxEnv(QuadrupedGo2MjxEnv):
    """Go2 trot for DIAL-MPC, with the full dial-mpc trot reward (env-reward mode).

    Same Go2 MJX transition as :class:`QuadrupedGo2MjxEnv` but at the DIAL control
    rate ``dt=0.02``, and it exposes ``reward(state, action, t)`` reproducing the
    dial-mpc go2 trot reward INCLUDING the gait z_feet term (foot-site forward
    kinematics via mjx) plus upright / yaw / body-vel / ang-vel / height tracking.
    The DIAL backend uses this env-reward path (time ``t`` advances with the
    receding-horizon step, so the gait phase is globally consistent). ``@dataclass``
    guarantees the ``dt`` / ``model`` overrides take effect.

    Foot order is the menagerie go2 order [FL, FR, RL, RR]; the trot gait phase is
    the diagonal pairing (FL,RR)=(FR,RL opposite). NOTE: act2tau PD control and a
    force-XML exact-parity variant are planned follow-ups.
    """

    dt: float = 0.02
    model: str = "go2"
    act_dim: int = 12
    control_limit: float = 1.0

    # trot reward params (dial-mpc go2 trot)
    _vx_tar: float = 0.8
    _gait_duty: float = 0.45
    _gait_cadence: float = 2.0
    _gait_amplitude: float = 0.08
    _foot_radius: float = 0.0175

    def __post_init__(self) -> None:
        super().__post_init__()
        import mujoco
        feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self._feet_site_id = jnp.array(
            [mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE.value, f) for f in feet]
        )
        assert not bool((self._feet_site_id < 0).any()), "go2 foot sites not found"
        # trot = diagonal gait in [FL, FR, RL, RR] order
        self._gait_phase = jnp.array([0.0, 0.5, 0.5, 0.0], dtype=jnp.float32)
        self._z_nom = float(self._home_qpos[2]) if self._home_qpos is not None else 0.3

        # act2tau PD setup (dial-mpc go2: kp=30, kd=0). joint_range in menagerie
        # order [FL, FR, RL, RR]; front calf [-2.3,-0.85], rear calf [-2.3,-1.3].
        self._kp, self._kd = 30.0, 0.0
        _hip, _thigh = [-0.5, 0.5], [0.4, 1.4]
        _cf, _cr = [-2.3, -0.85], [-2.3, -1.3]
        self._joint_range = jnp.array(
            [_hip, _thigh, _cf,  _hip, _thigh, _cf,
             _hip, _thigh, _cr,  _hip, _thigh, _cr], dtype=jnp.float32
        )
        self._phys_range = jnp.asarray(self._model.jnt_range[1:13], dtype=jnp.float32)
        self._default_pose = jnp.asarray(self._home_qpos[7:7 + 12], dtype=jnp.float32)

    def reward(self, state, action, t):
        """dial-mpc go2 trot reward (higher = better); ``t`` = absolute step."""
        qpos = state[: self.nq]
        qvel = state[self.nq: self.nq + self.nv]

        # gait z_feet term: foot-site z via forward kinematics from qpos
        data = mjx.make_data(self._mjx_model).replace(qpos=qpos)
        data = mjx.kinematics(self._mjx_model, data)
        foot_z = data.site_xpos[self._feet_site_id][:, 2] - self._foot_radius
        z_tar = _trot_foot_step(
            self._gait_duty, self._gait_cadence, self._gait_amplitude,
            self._gait_phase, t * self.dt,
        )
        r_gait = -jnp.sum(((z_tar - foot_z) / 0.05) ** 2)

        # body terms from base pose/vel
        z = qpos[2]
        qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
        vwx, vwy = qvel[0], qvel[1]
        wz = qvel[5]
        yaw = jnp.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        c, s = jnp.cos(yaw), jnp.sin(yaw)
        vbx, vby = c * vwx + s * vwy, -s * vwx + c * vwy
        r_vel = -((vbx - self._vx_tar) ** 2 + vby ** 2)
        r_ang = -(wz ** 2)
        r_up = -(qx * qx + qy * qy)
        r_yaw = -(yaw ** 2)
        r_h = -((z - self._z_nom) ** 2)

        return (0.1 * r_gait + 0.5 * r_up + 0.3 * r_yaw
                + 1.0 * r_vel + 1.0 * r_ang + 1.0 * r_h)

    def _act2joint(self, a_pad):
        """Normalized action [-1,1] -> joint position targets (dial act2joint)."""
        act_norm = (a_pad + 1.0) / 2.0
        jt = self._joint_range[:, 0] + act_norm * (self._joint_range[:, 1] - self._joint_range[:, 0])
        return jnp.clip(jt, self._phys_range[:, 0], self._phys_range[:, 1])

    def _compute_ctrl(self, qpos, qvel, a_pad, ctrl_base, ctrl_min, ctrl_max):
        """act2tau PD: tau = kp*(joint_target - q) - kd*qd (dial leg_control=torque)."""
        joint_target = self._act2joint(a_pad)
        q = qpos[7:7 + 12]
        qd = qvel[6:6 + 12]
        tau = self._kp * (joint_target - q) - self._kd * qd
        return jnp.clip(tau, ctrl_min, ctrl_max)
