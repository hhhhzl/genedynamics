"""
Humanoid MJX (MuJoCo XLA) environment for JAX-based planning.

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


def _resolve_humanoid_model_path(model: str) -> Optional[str]:
    """Resolve MJCF path via robot registry. Prefer MJX-compatible XML when available."""
    # MJX env: g1 must use the MJX-compatible scene (scene.xml has cylinder-mesh
    # contacts mjx rejects; scene_mjx.xml uses mjx-friendly colliders).
    if model == "g1":
        try:
            from genedynamics.robots.g1.assets import g1_scene_path
            p = g1_scene_path(prefer_mjx=True)
            if p:
                return p
        except Exception:
            pass
    if model in ("h1", "h1_loco", "h1_push_crate"):
        # H1 MJX scenes vendored from dial-mpc (mjx-compatible). h1 = full
        # (nq=26/nu=19); h1_loco = reduced legs+torso (nq=18/nu=11); h1_push_crate
        # = full H1 + a freejoint/slide crate body (nq=27) the robot pushes.
        try:
            from genedynamics.robots.registry import _get_h1_path
            scene = {
                "h1_loco": "mjx_scene_h1_loco.xml",
                "h1_push_crate": "mjx_scene_h1_push_crate.xml",
            }.get(model, "mjx_scene_h1_walk.xml")
            p = _get_h1_path(scene)
            if p:
                return p
        except Exception:
            pass
    try:
        from genedynamics.robots import get_robot_registry
        reg = get_robot_registry()
        path = reg.get_model_path("humanoid", model)
        if path:
            return path
    except Exception:
        pass
    if model == "humanoid":
        try:
            from genedynamics.robots.registry import _get_humanoid_path
            return _get_humanoid_path()
        except Exception:
            pass
    if model == "g1":
        try:
            from genedynamics.robots.registry import _get_g1_path
            path = _get_g1_path()
            if path:
                return path
            import os
            menagerie = os.environ.get("MUJOCO_MENAGERIE_PATH")
            if menagerie:
                for f in ("g1_mjx.xml", "g1.xml"):
                    p = Path(menagerie) / "unitree_g1" / f
                    if p.exists():
                        return str(p)
        except Exception:
            pass
    return None


@dataclass
class HumanoidMjxEnv:
    """
    Humanoid environment with MJX (MuJoCo XLA) physics backend.

    Provides JAX-compatible jax_transition/jax_model_transition for diffusion
    solvers (MBD, 2GO, etc.). State format: flat [qpos; qvel].
    Action: joint torques, clipped to [-control_limit, control_limit].

    Attributes:
        dt: Simulation timestep (s)
        horizon: Planning horizon (steps)
        act_dim: Action dimension (17 for humanoid, 23 for G1)
        control_limit: Torque limit (symmetric)
        target: Goal position (x, y, z)
        model: Robot model id ("humanoid", "g1", etc.)
        model_path: Override path to MJCF (optional)
        obstacles: Optional ObstacleManager for jax_sdf (Phase 5)
    """

    dt: float = 0.05
    horizon: int = 200
    act_dim: int = 17
    control_limit: float = 0.4
    target: Tuple[float, float, float] = (3.0, 0.0, 1.0)
    model: str = "humanoid"
    model_path: Optional[str] = None
    obstacles: Optional[Any] = None

    _model: Any = field(default=None, repr=False)
    _mjx_model: Any = field(default=None, repr=False)
    nq: int = 24
    nv: int = 23
    state_dim: int = 47
    nu: int = 17
    # physics substeps per control step: timestep = dt / _n_frames. >1 stabilises
    # stiff-contact bipeds (H1 with kp~200) that blow up at a single dt=0.02 step.
    _n_frames: int = 1

    def __post_init__(self) -> None:
        if not MJX_AVAILABLE:
            raise ImportError(
                "HumanoidMjxEnv requires mujoco, mujoco-mjx, and jax. "
                "Install with: pip install mujoco mujoco-mjx jax jaxlib"
            )
        path = self.model_path or _resolve_humanoid_model_path(self.model)
        if not path or not Path(path).exists():
            raise FileNotFoundError(
                f"Humanoid model not found for '{self.model}'. "
                f"Path: {path}. Install gymnasium for humanoid, or set MUJOCO_MENAGERIE_PATH for G1."
            )
        self._load_model(path)

    def _load_model(self, model_path: str) -> None:
        """Load MuJoCo model and create MJX device arrays."""
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._model.opt.timestep = self.dt / max(1, int(self._n_frames))
        # MJX only supports mjSOL_CG and mjSOL_NEWTON (not mjSOL_PGS)
        if self._model.opt.solver == mujoco.mjtSolver.mjSOL_PGS:
            self._model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        self._mjx_model = mjx.put_model(self._model)
        self.nq = self._model.nq
        self.nv = self._model.nv
        self.nu = self._model.nu
        self.act_dim = self.nu
        self.state_dim = self.nq + self.nv

    def reset(self, rng: Optional[Any] = None, seed: Optional[int] = None, **kwargs: Any) -> Tuple[np.ndarray, Dict]:
        """Reset to default standing pose (base z=1.0, identity quat)."""
        if seed is not None:
            np.random.seed(seed)
        data = mjx.make_data(self._mjx_model)
        qpos = np.asarray(data.qpos, dtype=np.float64).copy()
        qvel = np.asarray(data.qvel, dtype=np.float64).copy()
        if qpos.size >= 3:
            qpos[2] = 1.0
        if qpos.size >= 7:
            qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        flat = np.concatenate([qpos, qvel]).astype(np.float32)
        return flat, {}

    def transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """NumPy transition for execution. state/action -> next_state."""
        state = np.asarray(state, dtype=np.float64).ravel()
        action = np.asarray(action, dtype=np.float64).ravel()
        action = np.clip(action, -self.control_limit, self.control_limit)
        if action.size > self.nu:
            action = action[: self.nu]
        elif action.size < self.nu:
            action = np.pad(action, (0, self.nu - action.size), constant_values=0.0)

        data = mjx.make_data(self._mjx_model)
        data = data.replace(
            qpos=jnp.asarray(state[: self.nq], dtype=jnp.float64),
            qvel=jnp.asarray(state[self.nq : self.nq + self.nv], dtype=jnp.float64),
        )
        ctrl = jnp.zeros(self.nu, dtype=jnp.float64)
        ctrl = ctrl.at[:].set(jnp.asarray(action, dtype=jnp.float64))
        data = data.replace(ctrl=ctrl)
        next_data = mjx.step(self._mjx_model, data)
        next_state = np.concatenate([
            np.asarray(next_data.qpos, dtype=np.float32),
            np.asarray(next_data.qvel, dtype=np.float32),
        ])
        return next_state

    def jax_transition(self, state: Any, action: Any) -> Any:
        """
        JAX-compatible transition for planning. Pure function, supports vmap.
        """
        state = jnp.asarray(state, dtype=jnp.float64)
        action = jnp.asarray(action, dtype=jnp.float64)
        action = jnp.clip(action, -self.control_limit, self.control_limit)

        def _step(s: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
            a_flat = jnp.reshape(a, (-1))
            a_pad = jnp.concatenate([a_flat, jnp.zeros(self.nu, dtype=a_flat.dtype)])[: self.nu]
            data = mjx.make_data(self._mjx_model)
            data = data.replace(
                qpos=s[: self.nq],
                qvel=s[self.nq : self.nq + self.nv],
            )
            ctrl = self._compute_ctrl(s[: self.nq], s[self.nq : self.nq + self.nv], a_pad)
            data = data.replace(ctrl=ctrl)
            # n_frames physics substeps at timestep = dt/n_frames (hold ctrl)
            def _sub(d, _):
                return mjx.step(self._mjx_model, d), None
            next_data, _ = jax.lax.scan(_sub, data, None, length=max(1, int(self._n_frames)))
            next_state = jnp.concatenate([next_data.qpos, next_data.qvel])
            # Guard against mjx blow-ups (no clamps before): on NaN/inf or huge
            # magnitude, keep the configuration and zero the velocity so the
            # diffusion rollout stays finite (the sample just scores poorly).
            safe = jnp.concatenate([s[: self.nq], jnp.zeros(self.nv, dtype=s.dtype)])
            sane = jnp.logical_and(
                jnp.all(jnp.isfinite(next_state)),
                jnp.max(jnp.abs(next_state)) <= 1.0e3,
            )
            return jnp.where(sane, next_state, safe)

        if state.ndim == 1:
            return _step(state, action)
        return jax.vmap(_step)(state, action)

    def _compute_ctrl(self, qpos, qvel, a_pad):
        """Map the (padded) action to actuator ctrl. Default: action as direct
        ctrl. Subclasses (e.g. H1 loco) override for act2tau PD."""
        return a_pad

    def jax_model_transition(self, state: Any, action: Any) -> Any:
        """Same as jax_transition."""
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
class HumanoidSimplifiedMjxEnv(HumanoidMjxEnv):
    """Gymnasium humanoid with MJX."""

    model: str = "humanoid"


@dataclass
class HumanoidG1MjxEnv(HumanoidMjxEnv):
    """Unitree G1 humanoid with MJX (mujoco_menagerie).

    ``@dataclass`` required so ``model="g1"`` overrides the parent "humanoid"
    default (without it the inherited __init__ would silently load the humanoid).
    """

    model: str = "g1"
    act_dim: int = 23
    control_limit: float = 0.4


@dataclass
class HumanoidH1MjxEnv(HumanoidMjxEnv):
    """Unitree H1 humanoid with MJX (dial-mpc mjx scene, vendored, mjx-compatible).

    Full H1 (nq=26/nu=19). ``@dataclass`` so ``model="h1"`` takes effect.
    """

    model: str = "h1"
    act_dim: int = 19
    control_limit: float = 1.0
