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
    model: str = "ant"
    model_path: Optional[str] = None
    obstacles: Optional[Any] = None

    _model: Any = field(default=None, repr=False)
    _mjx_model: Any = field(default=None, repr=False)
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
        self.nq = self._model.nq
        self.nv = self._model.nv
        self.nu = self._model.nu
        self.act_dim = self.nu
        self.state_dim = self.nq + self.nv

    def reset(self, rng: Optional[Any] = None, seed: Optional[int] = None, **kwargs: Any) -> Tuple[np.ndarray, Dict]:
        """Reset to default standing pose (base z=0.5, identity quat)."""
        if seed is not None:
            np.random.seed(seed)
        data = mjx.make_data(self._mjx_model)
        qpos = np.asarray(data.qpos, dtype=np.float64).copy()
        qvel = np.asarray(data.qvel, dtype=np.float64).copy()
        if qpos.size >= 3:
            qpos[2] = 0.5
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

        Args:
            state: (nq+nv,) or (B, nq+nv) JAX array
            action: (nu,) or (B, nu) JAX array

        Returns:
            next_state: same shape as state
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
            data = data.replace(ctrl=a_pad)
            next_data = mjx.step(self._mjx_model, data)
            return jnp.concatenate([next_data.qpos, next_data.qvel])

        if state.ndim == 1:
            return _step(state, action)
        return jax.vmap(_step)(state, action)

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


class QuadrupedAntMjxEnv(QuadrupedMjxEnv):
    """Ant quadruped with MJX (gymnasium ant.xml)."""

    model: str = "ant"


class QuadrupedGo2MjxEnv(QuadrupedMjxEnv):
    """Unitree Go2 quadruped with MJX (mujoco_menagerie)."""

    model: str = "go2"
    act_dim: int = 12
    control_limit: float = 1.0
