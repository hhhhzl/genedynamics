"""
Simplified humanoid physics env using MuJoCo humanoid.

Provides transition, reset, cost interface.
Falls back to simple NumPy dynamics when MuJoCo unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    mujoco = None


def _resolve_humanoid_model_path(model: str) -> Optional[str]:
    """Resolve model path via robot registry or built-in."""
    try:
        from genedynamics.robots import get_robot_registry
        reg = get_robot_registry()
        path = reg.get_model_path("humanoid", model)
        if path:
            return path
    except Exception:
        pass
    if model in ("humanoid",):
        try:
            from genedynamics.robots.registry import _get_humanoid_path
            return _get_humanoid_path()
        except Exception:
            pass
    if model == "g1":
        try:
            from genedynamics.robots.registry import _get_g1_path
            return _get_g1_path() or None
        except Exception:
            pass
    return None


def _simple_humanoid_step(state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
    """Minimal placeholder: pos += vel*dt, vel += action[:3]*0.1."""
    s = np.asarray(state, dtype=np.float32).ravel()
    a = np.asarray(action, dtype=np.float32).ravel()
    nq, nv = 24, 23
    if s.size < nq + nv:
        s = np.pad(s, (0, nq + nv - s.size), mode="constant")
    pos = s[:3].copy()
    vel = s[nq : nq + 3].copy()
    vel += np.clip(a[:3] if a.size >= 3 else np.zeros(3), -0.4, 0.4) * 0.1
    vel[2] -= 0.5 * dt
    pos += vel * dt
    s[:3] = pos
    s[nq : nq + 3] = vel
    return s


@dataclass
class HumanoidBasePhysicsEnv:
    """
    Simplified humanoid env using MuJoCo humanoid.

    State: flat [qpos; qvel], humanoid: nq=24, nv=23
    Action: 17D torque
    """

    dt: float = 0.05
    horizon: int = 200
    act_dim: int = 17
    control_limit: float = 0.4
    target: Tuple[float, float, float] = (3.0, 0.0, 1.0)
    model_path: Optional[str] = None
    model: str = "humanoid"  # humanoid | h1 (registry)

    def __post_init__(self):
        self._backend = None
        self._model = None
        self._data = None
        self.nq = 24
        self.nv = 23
        self.state_dim = self.nq + self.nv

        # Override from registry if model differs
        try:
            from genedynamics.robots import get_robot_registry
            reg = get_robot_registry()
            entry = reg.get("humanoid", self.model)
            if entry and entry.model_id != "humanoid":
                self.nq, self.nv, self.act_dim = entry.nq, entry.nv, entry.act_dim
                self.state_dim = self.nq + self.nv
        except Exception:
            pass

        if MUJOCO_AVAILABLE:
            try:
                from genedynamics.core.backends.adapters.mujoco_adapter import MujocoPhysicsBackend
                path = self.model_path or _resolve_humanoid_model_path(self.model)
                if path:
                    self._backend = MujocoPhysicsBackend(model_path=path, dt=self.dt)
                    self._model = self._backend.model
                    self._data = self._backend.data
                    self.nq = self._model.nq
                    self.nv = self._model.nv
                    self.state_dim = self.nq + self.nv
                    self.act_dim = self._model.nu
            except Exception:
                import warnings
                warnings.warn(f"MuJoCo {self.model} load failed. Using simple NumPy fallback.", UserWarning)

    def _use_mujoco(self) -> bool:
        return self._backend is not None

    def reset(self, rng: Optional[Any] = None) -> Tuple[np.ndarray, Dict]:
        """Reset to initial state."""
        if self._use_mujoco():
            state = self._backend.reset()
            flat = np.concatenate([
                np.asarray(state["qpos"], dtype=np.float32),
                np.asarray(state["qvel"], dtype=np.float32),
            ])
        else:
            flat = np.zeros(self.nq + self.nv, dtype=np.float32)
            flat[2] = 1.0
            flat[7:11] = [1, 0, 0, 0]
        return flat, {}

    def transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Step dynamics."""
        state = np.asarray(state, dtype=np.float32).ravel()
        action = np.asarray(action, dtype=np.float32).ravel()
        action = np.clip(action, -self.control_limit, self.control_limit)

        if self._use_mujoco():
            if action.size > self._model.nu:
                action = action[: self._model.nu]
            self._backend.set_state(state)
            next_state = self._backend.step(action)
            return np.concatenate([
                np.asarray(next_state["qpos"], dtype=np.float32),
                np.asarray(next_state["qvel"], dtype=np.float32),
            ])
        return _simple_humanoid_step(state, action, self.dt)

    @property
    def nu(self) -> int:
        return self._model.nu if self._model is not None else self.act_dim

    def cost(self, state: np.ndarray) -> float:
        """Task cost: base position error + velocity penalty."""
        pos = state[:3]
        vel = state[self.nq : self.nq + 3] if state.size > self.nq + 3 else np.zeros(3)
        target = np.asarray(self.target, dtype=np.float32)
        pos_err = np.sum((pos - target) ** 2)
        vel_err = 0.1 * np.sum(vel ** 2)
        return float(pos_err + vel_err)


@dataclass
class HumanoidG1PhysicsEnv(HumanoidBasePhysicsEnv):
    """Unitree G1 humanoid (mujoco_menagerie when available).

    ``@dataclass`` required so ``model="g1"`` overrides the parent "humanoid"
    default.
    """

    model: str = "g1"
    act_dim: int = 23  # G1 29dof variant typically uses 23 controllable
