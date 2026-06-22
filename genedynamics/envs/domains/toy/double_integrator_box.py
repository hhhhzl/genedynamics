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
    from genedynamics.core.registry.environments import register_env
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    register_env = None

Array = np.ndarray


@dataclass
class DoubleIntegratorBoxEnv:
    dt: float = 0.1
    horizon: int = 80
    p_max: float = 2.0
    v_max: float = 2.0
    act_dim: int = 1
    target: float = 0.0
    vel_weight: float = 0.1
    control_limit: float = 1.0

    def reset(self, rng: Optional["jax.Array"] = None) -> Tuple[Array, dict]:
        """Sample an initial state. Accepts optional JAX rng for compatibility."""
        if rng is not None:
            if jax is None:
                raise RuntimeError("JAX is required to call reset with rng.")
            rng_p, rng_v = jax.random.split(rng)
            p0 = float(jax.random.uniform(rng_p, (), minval=-self.p_max, maxval=self.p_max))
            v0 = float(jax.random.uniform(rng_v, (), minval=-self.v_max, maxval=self.v_max))
        else:
            p0 = np.random.uniform(-self.p_max, self.p_max)
            v0 = np.random.uniform(-self.v_max, self.v_max)
        x = np.array([p0, v0], dtype=np.float32)
        info = {}
        return x, info

    # ------------------------------------------------------------------ helpers
    def _project_state(self, x_next: Array) -> Array:
        x_proj = np.asarray(x_next, dtype=np.float32).copy()
        x_proj[0] = np.clip(x_proj[0], -self.p_max, self.p_max)
        x_proj[1] = np.clip(x_proj[1], -self.v_max, self.v_max)
        return x_proj

    def cost(self, state: Array) -> float:
        pos_err = (state[0] - self.target) ** 2
        vel_err = self.vel_weight * (state[1] ** 2)
        return float(pos_err + vel_err)

    def jax_cost(self, state):
        if jnp is None:
            raise RuntimeError("jax_cost requires JAX to be installed.")
        pos_err = (state[..., 0] - self.target) ** 2
        vel_err = self.vel_weight * (state[..., 1] ** 2)
        return pos_err + vel_err

    # ---------------------------------------------------------------- simulation
    def step(self, x_next: Array, u: Array, t: int, info):
        """Project the proposed state, compute task cost, and advance time."""
        x_proj = self._project_state(x_next)
        cost = self.cost(x_proj)

        done = (t + 1) >= self.horizon
        info_n = {}
        return x_proj, cost, done, info_n

    def transition(self, state: Array, action: Array) -> Array:
        """
        Deterministic double-integrator transition using a control input.

        This is provided for diffusion planners that operate in action space.
        """
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        u = np.clip(action[0], -self.control_limit, self.control_limit)

        p, v = state
        v_next = v + self.dt * u
        p_next = p + self.dt * v_next

        return self._project_state(np.array([p_next, v_next], dtype=np.float32))

    def rollout_actions(self, state: Array, actions: Array) -> Array:
        """Roll out a sequence of control actions and return the resulting states."""
        x = np.asarray(state, dtype=np.float32)
        traj = [x]
        for act in np.asarray(actions, dtype=np.float32):
            x = self.transition(x, act)
            traj.append(x)
        return np.stack(traj, axis=0)

    def model_transition(self, state: Array, action: Array) -> Array:
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        u = np.clip(action[0], -self.control_limit, self.control_limit)

        p, v = state
        v_next = v + self.dt * u
        p_next = p + self.dt * v_next

        return np.array([p_next, v_next], dtype=np.float32)

    def jax_transition(self, state, action):
        """JAX-compatible transition used for differentiable planners."""
        if jnp is None:
            raise RuntimeError("jax_transition requires JAX to be installed.")
        u = jnp.clip(action[..., 0], -self.control_limit, self.control_limit)
        p = state[..., 0]
        v = state[..., 1]
        v_next = v + self.dt * u
        p_next = p + self.dt * v_next
        p_next = jnp.clip(p_next, -self.p_max, self.p_max)
        v_next = jnp.clip(v_next, -self.v_max, self.v_max)
        return jnp.stack([p_next, v_next], axis=-1)

    def jax_model_transition(self, state, action):
        if jnp is None:
            raise RuntimeError("jax_model_transition requires JAX to be installed.")
        u = jnp.clip(action[..., 0], -self.control_limit, self.control_limit)
        p = state[..., 0]
        v = state[..., 1]
        v_next = v + self.dt * u
        p_next = p + self.dt * v_next
        return jnp.stack([p_next, v_next], axis=-1)

    def jax_env_transition(self, state, action):
        if jnp is None:
            raise RuntimeError("jax_env_transition requires JAX to be installed.")
        u = jnp.clip(action[..., 0], -self.control_limit, self.control_limit)
        p = state[..., 0]
        v = state[..., 1]
        v_next = v + self.dt * u
        p_next = p + self.dt * v_next
        p_next = jnp.clip(p_next, -self.p_max, self.p_max)
        v_next = jnp.clip(v_next, -self.v_max, self.v_max)
        return jnp.stack([p_next, v_next], axis=-1)


# Register environment to registry
if REGISTRY_AVAILABLE and register_env is not None:
    register_env("double_integrator_box", DoubleIntegratorBoxEnv)
