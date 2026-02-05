from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - JAX is optional at runtime
    jax = None
    jnp = None

# Register environment to registry
try:
    from enerdynamics.core.registry.environments import register_env
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    register_env = None

Array = np.ndarray


@dataclass
class SingleIntegratorBox2DEnv:
    """
    Single-integrator point robot in a 2D box.

    State:  x = [px, py]
    Action: u = [vx, vy] (interpreted as velocity; clipped by control_limit)

    Transition:
        x_{t+1} = clip_box(x_t + dt * clip(u))
    """

    dt: float = 0.1
    horizon: int = 80
    p_max: float = 2.0
    act_dim: int = 2
    target: Tuple[float, float] = (0.0, 0.0)
    control_limit: float = 1.0

    def reset(self, rng: Optional["jax.Array"] = None) -> Tuple[Array, dict]:
        if rng is None:
            p0 = np.random.uniform(-self.p_max, self.p_max, size=2)
        else:
            if jax is None:
                raise RuntimeError("JAX is required to call reset with rng.")
            p0 = np.array(
                jax.random.uniform(rng, (2,), minval=-self.p_max, maxval=self.p_max),
                dtype=np.float32,
            )
        x = np.asarray(p0, dtype=np.float32)
        info = {}
        return x, info

    def _project_state(self, x_next: Array) -> Array:
        x_proj = np.asarray(x_next, dtype=np.float32).copy()
        x_proj[0:2] = np.clip(x_proj[0:2], -self.p_max, self.p_max)
        return x_proj

    def cost(self, state: Array) -> float:
        pos = np.asarray(state, dtype=np.float32)[0:2]
        target = np.asarray(self.target, dtype=np.float32)
        pos_err = np.sum((pos - target) ** 2)
        return float(pos_err)

    def jax_cost(self, state):
        if jnp is None:
            raise RuntimeError("jax_cost requires JAX to be installed.")
        target = jnp.asarray(self.target, dtype=jnp.float32)
        pos = state[..., 0:2]
        return jnp.sum((pos - target) ** 2, axis=-1)

    def step(self, x_next: Array, u: Array, t: int, info):
        x_proj = self._project_state(x_next)
        cost = self.cost(x_proj)
        done = (t + 1) >= self.horizon
        info_n = {}
        return x_proj, cost, done, info_n

    def transition(self, state: Array, action: Array) -> Array:
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        u = np.clip(action, -self.control_limit, self.control_limit)
        p_next = state[0:2] + self.dt * u[0:2]
        return self._project_state(p_next.astype(np.float32))

    def rollout_actions(self, state: Array, actions: Array) -> Array:
        x = np.asarray(state, dtype=np.float32)
        traj = [x]
        for act in np.asarray(actions, dtype=np.float32):
            x = self.transition(x, act)
            traj.append(x)
        return np.stack(traj, axis=0)

    def model_transition(self, state: Array, action: Array) -> Array:
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        u = np.clip(action, -self.control_limit, self.control_limit)
        p_next = state[0:2] + self.dt * u[0:2]
        return p_next.astype(np.float32)

    def jax_transition(self, state, action):
        if jnp is None:
            raise RuntimeError("jax_transition requires JAX to be installed.")
        state = jnp.atleast_1d(jnp.asarray(state, dtype=jnp.float32))
        action = jnp.atleast_1d(jnp.asarray(action, dtype=jnp.float32))
        u = jnp.clip(action, -self.control_limit, self.control_limit)
        p_next = state[..., 0:2] + self.dt * u[..., 0:2]
        p_next = jnp.clip(p_next, -self.p_max, self.p_max)
        return p_next

    def jax_model_transition(self, state, action):
        # For single-integrator, model and env transitions are identical.
        return self.jax_transition(state, action)

    def jax_env_transition(self, state, action):
        return self.jax_transition(state, action)

    def jax_sdf(self, pos):
        """
        Simple JAX SDF to the box boundary: min distance to walls.

        Args:
            pos: (..., 2) positions
        Returns:
            (...,) signed distance (positive inside box, zero on wall, negative outside)
        """
        if jnp is None:
            raise RuntimeError("jax_sdf requires JAX to be installed.")
        pos = jnp.asarray(pos, dtype=jnp.float32)
        # Distance to each wall (positive inside)
        dist_pos = self.p_max - pos
        dist_neg = self.p_max + pos
        dist_all = jnp.stack([dist_pos[..., 0], dist_pos[..., 1], dist_neg[..., 0], dist_neg[..., 1]], axis=-1)
        return jnp.min(dist_all, axis=-1)


# Register environment to registry
if REGISTRY_AVAILABLE and register_env is not None:
    register_env("single_integrator_box_2d", SingleIntegratorBox2DEnv)

