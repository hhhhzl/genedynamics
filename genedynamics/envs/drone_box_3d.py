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
class DroneBox3DEnv:
    """
    3D quadrotor/drone environment with simplified double integrator dynamics.
    
    State: x = [px, py, pz, vx, vy, vz]
    Action: u = [ax, ay, az] (acceleration commands, clipped by control_limit)
    
    Transition:
        v_{t+1} = clip(v_t + dt * clip(u))
        p_{t+1} = clip_box(p_t + dt * v_{t+1})
    
    Note: This is a simplified model. For full quadrotor dynamics,
    use the DroneModel class with proper motor control.
    """

    dt: float = 0.1
    horizon: int = 80
    p_max: float = 2.0
    v_max: float = 2.0
    act_dim: int = 3
    target: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    vel_weight: float = 0.1
    control_limit: float = 1.0

    def reset(self, rng: Optional["jax.Array"] = None) -> Tuple[Array, dict]:
        """Sample an initial state. Accepts optional JAX rng for compatibility."""
        if rng is None:
            p0 = np.random.uniform(-self.p_max, self.p_max, size=3)
            v0 = np.random.uniform(-self.v_max, self.v_max, size=3)
        else:
            if jax is None:
                raise RuntimeError("JAX is required to call reset with rng.")
            rng_p, rng_v = jax.random.split(rng)
            p0 = np.array(
                jax.random.uniform(rng_p, (3,), minval=-self.p_max, maxval=self.p_max),
                dtype=np.float32,
            )
            v0 = np.array(
                jax.random.uniform(rng_v, (3,), minval=-self.v_max, maxval=self.v_max),
                dtype=np.float32,
            )
        x = np.concatenate([p0, v0]).astype(np.float32)
        info = {}
        return x, info

    def _project_state(self, x_next: Array) -> Array:
        """Project state to valid bounds."""
        x_proj = np.asarray(x_next, dtype=np.float32).copy()
        x_proj[0:3] = np.clip(x_proj[0:3], -self.p_max, self.p_max)
        x_proj[3:6] = np.clip(x_proj[3:6], -self.v_max, self.v_max)
        return x_proj

    def cost(self, state: Array) -> float:
        """Compute cost for a given state."""
        pos = state[0:3]
        vel = state[3:6]
        target = np.asarray(self.target, dtype=np.float32)
        pos_err = np.sum((pos - target) ** 2)
        vel_err = self.vel_weight * np.sum(vel ** 2)
        return float(pos_err + vel_err)

    def jax_cost(self, state):
        """JAX-compatible cost function."""
        if jnp is None:
            raise RuntimeError("jax_cost requires JAX to be installed.")
        target = jnp.asarray(self.target, dtype=jnp.float32)
        pos = state[..., 0:3]
        vel = state[..., 3:6]
        pos_err = jnp.sum((pos - target) ** 2, axis=-1)
        vel_err = self.vel_weight * jnp.sum(vel ** 2, axis=-1)
        return pos_err + vel_err

    def step(self, x_next: Array, u: Array, t: int, info):
        """Project the proposed state, compute task cost, and advance time."""
        x_proj = self._project_state(x_next)
        cost = self.cost(x_proj)
        done = (t + 1) >= self.horizon
        info_n = {}
        return x_proj, cost, done, info_n

    def transition(self, state: Array, action: Array) -> Array:
        """
        Deterministic double-integrator transition in 3D.
        
        Args:
            state: Current state [px, py, pz, vx, vy, vz]
            action: Acceleration command [ax, ay, az]
            
        Returns:
            Next state
        """
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        u = np.clip(action, -self.control_limit, self.control_limit)
        
        p = state[0:3]
        v = state[3:6]
        v_next = v + self.dt * u
        p_next = p + self.dt * v_next
        
        return self._project_state(np.concatenate([p_next, v_next]).astype(np.float32))

    def rollout_actions(self, state: Array, actions: Array) -> Array:
        """Roll out a sequence of control actions and return the resulting states."""
        x = np.asarray(state, dtype=np.float32)
        traj = [x]
        for act in np.asarray(actions, dtype=np.float32):
            x = self.transition(x, act)
            traj.append(x)
        return np.stack(traj, axis=0)

    def model_transition(self, state: Array, action: Array) -> Array:
        """
        Model transition (without state projection/clipping).
        
        Used by planners that want to simulate without bounds.
        """
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        u = np.clip(action, -self.control_limit, self.control_limit)
        
        p = state[0:3]
        v = state[3:6]
        v_next = v + self.dt * u
        p_next = p + self.dt * v_next
        
        return np.concatenate([p_next, v_next]).astype(np.float32)

    def jax_transition(self, state, action):
        """JAX-compatible transition with state clipping."""
        if jnp is None:
            raise RuntimeError("jax_transition requires JAX to be installed.")
        u = jnp.clip(action, -self.control_limit, self.control_limit)
        p = state[..., 0:3]
        v = state[..., 3:6]
        v_next = v + self.dt * u
        p_next = p + self.dt * v_next
        p_next = jnp.clip(p_next, -self.p_max, self.p_max)
        v_next = jnp.clip(v_next, -self.v_max, self.v_max)
        return jnp.concatenate([p_next, v_next], axis=-1)

    def jax_model_transition(self, state, action):
        """JAX-compatible model transition (without clipping)."""
        if jnp is None:
            raise RuntimeError("jax_model_transition requires JAX to be installed.")
        u = jnp.clip(action, -self.control_limit, self.control_limit)
        p = state[..., 0:3]
        v = state[..., 3:6]
        v_next = v + self.dt * u
        p_next = p + self.dt * v_next
        return jnp.concatenate([p_next, v_next], axis=-1)

    def jax_env_transition(self, state, action):
        """JAX-compatible environment transition (same as jax_transition)."""
        if jnp is None:
            raise RuntimeError("jax_env_transition requires JAX to be installed.")
        return self.jax_transition(state, action)


# Register environment to registry
if REGISTRY_AVAILABLE and register_env is not None:
    register_env("drone_box_3d", DroneBox3DEnv)

