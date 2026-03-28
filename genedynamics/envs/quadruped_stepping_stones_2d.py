from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import jax.numpy as jnp
except Exception:
    jnp = None

from genedynamics.core.energy import EnergyTerm, LegacyEnergyFunctional
from genedynamics.tasks.stepping_stones import SteppingStonesScene, sample_stepping_stones_scene


def _to_np(x: Any, n: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size < n:
        arr = np.pad(arr, (0, n - arr.size), constant_values=0.0)
    return arr[:n]


@dataclass
class QuadrupedSteppingStones2DEnv:
    """
    Kinematic biped stepping abstraction for quadruped footstep planning.

    State  : [pL_x, pL_y, pR_x, pR_y]
    Action : [dL_x, dL_y, dR_x, dR_y]
    """

    dt: float = 1.0
    horizon: int = 12
    control_limit: float = 0.45
    l_max: float = 0.35
    stance_width: float = 0.30
    start_mid: Tuple[float, float] = (-1.25, 0.0)
    goal_mid: Tuple[float, float] = (1.25, 0.0)
    obstacles: Optional[Any] = None
    scene_seed: int = 0
    scene_level: int = 1
    scene: Optional[SteppingStonesScene] = None

    act_dim: int = 4
    state_dim: int = 4

    def __post_init__(self) -> None:
        scene = self.scene
        if scene is None and self.obstacles is not None and hasattr(self.obstacles, "stepping_scene"):
            scene = getattr(self.obstacles, "stepping_scene")
        if scene is None:
            scene = sample_stepping_stones_scene(
                level=int(self.scene_level),
                seed=int(self.scene_seed),
                l_max=float(self.l_max),
                stance_width=float(self.stance_width),
                start_mid=self.start_mid,
                goal_mid=self.goal_mid,
            )
        self.scene = scene
        self.horizon = int(scene.k_horizon)
        self.l_max = float(scene.l_max)
        self.control_limit = max(float(self.control_limit), float(scene.l_max))
        self.target = np.asarray(scene.goal_mid, dtype=np.float32)
        self._default_state = np.asarray(
            [
                scene.start_left[0],
                scene.start_left[1],
                scene.start_right[0],
                scene.start_right[1],
            ],
            dtype=np.float32,
        )

    def reset(self, rng: Optional[Any] = None, seed: Optional[int] = None, **kwargs: Any):
        _ = (rng, seed, kwargs)
        return self._default_state.copy(), {}

    def _clip_step(self, dxy: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(dxy))
        if n <= self.l_max or n < 1e-8:
            return dxy
        return (self.l_max / n) * dxy

    def transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x = _to_np(state, 4)
        u = _to_np(action, 4)
        u = np.clip(u, -self.control_limit, self.control_limit)
        d_l = self._clip_step(self.dt * u[:2])
        d_r = self._clip_step(self.dt * u[2:4])
        next_state = np.array(
            [x[0] + d_l[0], x[1] + d_l[1], x[2] + d_r[0], x[3] + d_r[1]],
            dtype=np.float32,
        )
        return next_state

    def rollout_actions(self, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        x = _to_np(state, 4)
        traj = [x.copy()]
        for a in np.asarray(actions, dtype=np.float32):
            x = self.transition(x, a)
            traj.append(x.copy())
        return np.asarray(traj, dtype=np.float32)

    def model_transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self.transition(state, action)

    def jax_transition(self, state: Any, action: Any) -> Any:
        if jnp is None:
            raise RuntimeError("jax_transition requires JAX.")
        x = jnp.asarray(state, dtype=jnp.float32).reshape(-1)
        u = jnp.asarray(action, dtype=jnp.float32).reshape(-1)
        u = jnp.clip(u[:4], -self.control_limit, self.control_limit)

        def _clip_norm(v):
            n = jnp.linalg.norm(v)
            scale = jnp.where(n > self.l_max, self.l_max / jnp.maximum(n, 1e-8), 1.0)
            return v * scale

        d_l = _clip_norm(self.dt * u[:2])
        d_r = _clip_norm(self.dt * u[2:4])
        return jnp.asarray([x[0] + d_l[0], x[1] + d_l[1], x[2] + d_r[0], x[3] + d_r[1]], dtype=jnp.float32)

    def jax_model_transition(self, state: Any, action: Any) -> Any:
        return self.jax_transition(state, action)

    def cost(self, state: np.ndarray) -> float:
        x = _to_np(state, 4)
        mid = 0.5 * np.array([x[0] + x[2], x[1] + x[3]], dtype=np.float32)
        return float(np.sum((mid - self.target) ** 2))


def make_stepping_stones_energy(env: QuadrupedSteppingStones2DEnv) -> LegacyEnergyFunctional:
    scene = env.scene
    centers = np.asarray(scene.stones_centers, dtype=np.float32)
    radii = np.asarray(scene.stones_radii, dtype=np.float32)
    l_max = float(scene.l_max)
    goal_mid = np.asarray(scene.goal_mid, dtype=np.float32)

    if jnp is not None:
        centers_j = jnp.asarray(centers, dtype=jnp.float32)
        radii_j = jnp.asarray(radii, dtype=jnp.float32)
        goal_mid_j = jnp.asarray(goal_mid, dtype=jnp.float32)

        def _stone_violation(p):
            d = jnp.linalg.norm(p[None, :] - centers_j, axis=-1)
            margin = radii_j - d
            return jnp.maximum(0.0, -jnp.max(margin))

        def task_energy(x, u, ctx):
            _ = ctx
            x = jnp.asarray(x, dtype=jnp.float32).reshape(-1)
            p_l = x[:2]
            p_r = x[2:4]
            mid = 0.5 * (p_l + p_r)
            return jnp.sum((mid - goal_mid_j) ** 2)

        def smooth_energy(x, u, ctx):
            _ = (x, ctx)
            u = jnp.asarray(u, dtype=jnp.float32).reshape(-1)
            return 0.25 * jnp.sum(u ** 2)

        def foothold_energy(x, u, ctx):
            _ = (u, ctx)
            x = jnp.asarray(x, dtype=jnp.float32).reshape(-1)
            p_l = x[:2]
            p_r = x[2:4]
            return 12.0 * (_stone_violation(p_l) + _stone_violation(p_r))

        def step_bound_energy(x, u, ctx):
            _ = (x, ctx)
            u = jnp.asarray(u, dtype=jnp.float32).reshape(-1)
            s_l = jnp.maximum(0.0, jnp.linalg.norm(u[:2]) - l_max)
            s_r = jnp.maximum(0.0, jnp.linalg.norm(u[2:4]) - l_max)
            return 6.0 * (s_l ** 2 + s_r ** 2)

    else:
        def _stone_violation_np(p):
            d = np.linalg.norm(p[None, :] - centers, axis=-1)
            margin = radii - d
            return max(0.0, -float(np.max(margin)))

        def task_energy(x, u, ctx):
            _ = (u, ctx)
            x = np.asarray(x, dtype=np.float32).reshape(-1)
            p_l = x[:2]
            p_r = x[2:4]
            mid = 0.5 * (p_l + p_r)
            return float(np.sum((mid - goal_mid) ** 2))

        def smooth_energy(x, u, ctx):
            _ = (x, ctx)
            u = np.asarray(u, dtype=np.float32).reshape(-1)
            return 0.25 * float(np.sum(u ** 2))

        def foothold_energy(x, u, ctx):
            _ = (u, ctx)
            x = np.asarray(x, dtype=np.float32).reshape(-1)
            return 12.0 * (_stone_violation_np(x[:2]) + _stone_violation_np(x[2:4]))

        def step_bound_energy(x, u, ctx):
            _ = (x, ctx)
            u = np.asarray(u, dtype=np.float32).reshape(-1)
            s_l = max(0.0, float(np.linalg.norm(u[:2])) - l_max)
            s_r = max(0.0, float(np.linalg.norm(u[2:4])) - l_max)
            return 6.0 * (s_l * s_l + s_r * s_r)

    return LegacyEnergyFunctional(
        {
            "task": EnergyTerm(task_energy, 1.0),
            "smooth": EnergyTerm(smooth_energy, 1.0),
            "foothold": EnergyTerm(foothold_energy, 1.0),
            "step_bound": EnergyTerm(step_bound_energy, 1.0),
        }
    )

