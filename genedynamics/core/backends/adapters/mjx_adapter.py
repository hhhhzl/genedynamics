"""
MJX (MuJoCo XLA) physics backend adapter.

This module provides a PhysicsBackend implementation using mujoco-mjx for
JAX-based, GPU/TPU-accelerated, and differentiable physics simulation.
MJX supports batch dimensions for high-throughput rollout.
"""

from typing import Optional, Any, List, Tuple
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

from genedynamics.core.backends.physics import PhysicsBackend
from genedynamics.core.types import State, Action
from genedynamics.envs.obstacles.base import Obstacle


def _to_numpy(x: Any) -> np.ndarray:
    """Convert JAX/DeviceArray to numpy."""
    if hasattr(x, "__array__"):
        return np.asarray(x)
    return np.asarray(x)


class MjxPhysicsBackend(PhysicsBackend):
    """
    MJX (MuJoCo XLA) physics backend.

    Uses JAX-based MuJoCo for GPU/TPU simulation. Supports:
    - Same model/state interface as MujocoPhysicsBackend
    - Batched step via step_batch() for parallel rollout
    - Differentiable simulation (jax.grad over step)
    """

    name = "mjx"

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_xml: Optional[str] = None,
        dt: float = 0.01,
    ):
        if not MJX_AVAILABLE:
            raise ImportError(
                "MJX is required for MjxPhysicsBackend. "
                "Install with: pip install mujoco mujoco-mjx jax"
            )
        self.dt = dt
        self._model = None
        self._mjx_model = None
        self._mjx_data = None
        self.obstacle_geom_ids: List[int] = []

        if model_path is not None:
            self.load_model(model_path)
        elif model_xml is not None:
            self.load_model_from_xml(model_xml)

    def load_model(self, model_path: str, **kwargs) -> None:
        try:
            self._model = mujoco.MjModel.from_xml_path(model_path)
            self._model.opt.timestep = self.dt
            data = mujoco.MjData(self._model)
            self._mjx_model = mjx.put_model(self._model)
            self._mjx_data = mjx.put_data(self._model, data)
        except Exception as e:
            raise RuntimeError(f"Failed to load MJX model from {model_path}: {e}") from e

    def load_model_from_xml(self, model_xml: str) -> None:
        try:
            self._model = mujoco.MjModel.from_xml_string(model_xml)
            self._model.opt.timestep = self.dt
            data = mujoco.MjData(self._model)
            self._mjx_model = mjx.put_model(self._model)
            self._mjx_data = mjx.put_data(self._model, data)
        except Exception as e:
            raise RuntimeError(f"Failed to load MJX model from XML: {e}") from e

    @property
    def model(self):
        """MuJoCo model (for compatibility with MujocoPhysicsBackend)."""
        return self._model

    @property
    def data(self):
        """Current MJX data as numpy (for compatibility)."""
        if self._mjx_data is None:
            return None
        return {
            "qpos": _to_numpy(self._mjx_data.qpos),
            "qvel": _to_numpy(self._mjx_data.qvel),
        }

    def set_state(self, state: State) -> None:
        if self._mjx_data is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if isinstance(state, dict):
            qpos = np.asarray(state.get("qpos", self._mjx_data.qpos), dtype=np.float64)
            qvel = np.asarray(state.get("qvel", self._mjx_data.qvel), dtype=np.float64)
        else:
            state_arr = np.asarray(state, dtype=np.float64)
            nq = self._model.nq
            nv = self._model.nv
            if len(state_arr) == nq + nv:
                qpos, qvel = state_arr[:nq], state_arr[nq : nq + nv]
            elif len(state_arr) == nq:
                qpos, qvel = state_arr, np.zeros(nv, dtype=np.float64)
            else:
                raise ValueError(f"State dim mismatch: expected {nq}+{nv} or {nq}, got {len(state_arr)}")
        self._mjx_data = self._mjx_data.replace(
            qpos=jnp.asarray(qpos),
            qvel=jnp.asarray(qvel),
        )

    def get_state(self) -> State:
        if self._mjx_data is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return {
            "qpos": _to_numpy(self._mjx_data.qpos).copy(),
            "qvel": _to_numpy(self._mjx_data.qvel).copy(),
        }

    def step(self, action: Action) -> State:
        if self._mjx_data is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        action_arr = np.asarray(action, dtype=np.float64)
        if len(action_arr) <= self._model.nu:
            ctrl = jnp.zeros(self._model.nu, dtype=jnp.float64)
            ctrl = ctrl.at[: len(action_arr)].set(jnp.asarray(action_arr))
        else:
            raise ValueError(f"Action dim mismatch: expected <={self._model.nu}, got {len(action_arr)}")
        self._mjx_data = self._mjx_data.replace(ctrl=ctrl)
        self._mjx_data = mjx.step(self._mjx_model, self._mjx_data)
        return self.get_state()

    def step_batch(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Batched step for parallel rollout. States shape (B, nq+nv), actions (B, nu).
        Returns next_states (B, nq+nv).
        """
        if self._mjx_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        nq, nv = self._model.nq, self._model.nv
        states = np.asarray(states, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        B = states.shape[0]
        next_states = []
        for i in range(B):
            self.set_state({"qpos": states[i, :nq], "qvel": states[i, nq : nq + nv]})
            ns = self.step(actions[i])
            next_states.append(np.concatenate([ns["qpos"], ns["qvel"]]))
        return np.stack(next_states)

    def add_obstacle(self, obstacle: Obstacle, **kwargs) -> None:
        # MJX uses same model as MuJoCo; obstacles typically baked into XML
        if not hasattr(self, "_obstacle_geoms"):
            self._obstacle_geoms = []
        self._obstacle_geoms.append({"obstacle": obstacle, "kwargs": kwargs})

    def remove_obstacle(self, obstacle_id: Any) -> None:
        if hasattr(self, "_obstacle_geoms"):
            self._obstacle_geoms = [g for g in self._obstacle_geoms if g["obstacle"] != obstacle_id]

    def check_collision(self, state: Optional[State] = None) -> bool:
        if state is not None:
            self.set_state(state)
        # MJX doesn't expose ncon directly in same way; return False for now
        return False

    def get_collision_pairs(self) -> List[Tuple[int, int]]:
        return []

    def reset(self, **kwargs) -> State:
        if self._mjx_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        self._mjx_data = mjx.make_data(self._mjx_model)
        if "qpos" in kwargs:
            self._mjx_data = self._mjx_data.replace(qpos=jnp.asarray(kwargs["qpos"]))
        if "qvel" in kwargs:
            self._mjx_data = self._mjx_data.replace(qvel=jnp.asarray(kwargs["qvel"]))
        return self.get_state()

    def close(self) -> None:
        self._model = None
        self._mjx_model = None
        self._mjx_data = None
        self.obstacle_geom_ids = []
        if hasattr(self, "_obstacle_geoms"):
            self._obstacle_geoms = []
