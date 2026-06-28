from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - JAX is optional at runtime
    jax = None
    jnp = None

from genedynamics.envs.robots.drone import DroneModel

# Register environment to registry
try:
    from genedynamics.core.registry.environments import register_env
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    register_env = None

Array = np.ndarray


@dataclass
class DroneFull3DEnv:
    """
    Full 3D quadrotor environment with complete dynamics.
    
    State: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz] (12D)
    Action: [T1, T2, T3, T4] (4 motor thrusts, normalized 0-1)
    
    Uses DroneModel for physics simulation.
    """
    
    dt: float = 0.1
    horizon: int = 80
    p_max: float = 2.0
    v_max: float = 2.0
    act_dim: int = 4  # 4 motor thrusts
    target: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    vel_weight: float = 0.1
    orientation_weight: float = 0.1
    control_limit: float = 1.0  # Max motor thrust (normalized)
    
    # DroneModel parameters
    mass: float = 0.5
    Ixx: float = 0.0023
    Iyy: float = 0.0023
    Izz: float = 0.0046
    arm_length: float = 0.17
    kf: float = 3.16e-10
    km: float = 7.94e-12
    gravity: float = 9.81
    
    def __post_init__(self):
        """Initialize drone model after dataclass initialization."""
        self.drone_model = DroneModel(
            mass=self.mass,
            Ixx=self.Ixx,
            Iyy=self.Iyy,
            Izz=self.Izz,
            arm_length=self.arm_length,
            kf=self.kf,
            km=self.km,
            gravity=self.gravity,
        )
    
    def reset(self, rng: Optional["jax.Array"] = None) -> Tuple[Array, dict]:
        """Sample an initial state."""
        if rng is None:
            # Position
            p0 = np.random.uniform(-self.p_max, self.p_max, size=3)
            # Velocity
            v0 = np.random.uniform(-self.v_max, self.v_max, size=3)
            # Orientation (small random angles)
            euler0 = np.random.uniform(-0.1, 0.1, size=3)
            # Angular velocity
            ang_vel0 = np.random.uniform(-0.5, 0.5, size=3)
        else:
            if jax is None:
                raise RuntimeError("JAX is required to call reset with rng.")
            rng_p, rng_v, rng_e, rng_w = jax.random.split(rng, 4)
            p0 = np.array(
                jax.random.uniform(rng_p, (3,), minval=-self.p_max, maxval=self.p_max),
                dtype=np.float32,
            )
            v0 = np.array(
                jax.random.uniform(rng_v, (3,), minval=-self.v_max, maxval=self.v_max),
                dtype=np.float32,
            )
            euler0 = np.array(
                jax.random.uniform(rng_e, (3,), minval=-0.1, maxval=0.1),
                dtype=np.float32,
            )
            ang_vel0 = np.array(
                jax.random.uniform(rng_w, (3,), minval=-0.5, maxval=0.5),
                dtype=np.float32,
            )
        
        x = np.concatenate([p0, v0, euler0, ang_vel0]).astype(np.float32)
        info = {}
        return x, info
    
    def _project_state(self, x_next: Array) -> Array:
        """Project state to valid bounds."""
        x_proj = np.asarray(x_next, dtype=np.float32).copy()
        # Position bounds
        x_proj[0:3] = np.clip(x_proj[0:3], -self.p_max, self.p_max)
        # Velocity bounds
        x_proj[3:6] = np.clip(x_proj[3:6], -self.v_max, self.v_max)
        # Orientation bounds (Euler angles)
        x_proj[6:9] = np.clip(x_proj[6:9], -np.pi, np.pi)
        x_proj[8] = np.clip(x_proj[8], -np.pi/2, np.pi/2)  # Pitch
        # Angular velocity bounds
        x_proj[9:12] = np.clip(x_proj[9:12], -5.0, 5.0)
        return x_proj
    
    def cost(self, state: Array) -> float:
        """Compute cost for a given state."""
        pos = state[0:3]
        vel = state[3:6]
        euler = state[6:9]
        target = np.asarray(self.target, dtype=np.float32)
        
        pos_err = np.sum((pos - target) ** 2)
        vel_err = self.vel_weight * np.sum(vel ** 2)
        orientation_err = self.orientation_weight * np.sum(euler ** 2)
        
        return float(pos_err + vel_err + orientation_err)
    
    def jax_cost(self, state):
        """JAX-compatible cost function."""
        if jnp is None:
            raise RuntimeError("jax_cost requires JAX to be installed.")
        target = jnp.asarray(self.target, dtype=jnp.float32)
        pos = state[..., 0:3]
        vel = state[..., 3:6]
        euler = state[..., 6:9]
        
        pos_err = jnp.sum((pos - target) ** 2, axis=-1)
        vel_err = self.vel_weight * jnp.sum(vel ** 2, axis=-1)
        orientation_err = self.orientation_weight * jnp.sum(euler ** 2, axis=-1)
        
        return pos_err + vel_err + orientation_err
    
    def step(self, x_next: Array, u: Array, t: int, info):
        """Project the proposed state, compute task cost, and advance time."""
        x_proj = self._project_state(x_next)
        cost = self.cost(x_proj)
        done = (t + 1) >= self.horizon
        info_n = {}
        return x_proj, cost, done, info_n
    
    def transition(self, state: Array, action: Array) -> Array:
        """
        Deterministic quadrotor transition using motor thrusts.
        
        Args:
            state: Current state [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
            action: Motor thrusts [T1, T2, T3, T4] (normalized 0-1)
            
        Returns:
            Next state
        """
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        
        # Clip motor thrusts to [0, 1]
        motor_thrusts = np.clip(action, 0.0, self.control_limit)
        
        # Use DroneModel's step method
        next_state = self.drone_model.step(state, motor_thrusts, dt=self.dt)
        
        return self._project_state(next_state)
    
    def rollout_actions(self, state: Array, actions: Array) -> Array:
        """Roll out a sequence of control actions."""
        x = np.asarray(state, dtype=np.float32)
        traj = [x]
        for act in np.asarray(actions, dtype=np.float32):
            x = self.transition(x, act)
            traj.append(x)
        return np.stack(traj, axis=0)
    
    def model_transition(self, state: Array, action: Array) -> Array:
        """Model transition without state projection."""
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        motor_thrusts = np.clip(action, 0.0, self.control_limit)
        return self.drone_model.step(state, motor_thrusts, dt=self.dt)
    
    def jax_transition(self, state, action):
        """
        JAX-compatible transition.
        
        Note: DroneModel.step is not JAX-compatible, so we use a fallback
        that converts to numpy, computes, and converts back.
        """
        if jnp is None:
            raise RuntimeError("jax_transition requires JAX to be installed.")
        
        # Convert to numpy for computation
        state_np = np.asarray(state)
        action_np = np.asarray(action)
        
        # Handle batched and single inputs
        if state_np.ndim > 1:
            # Batch processing
            results = []
            for s, a in zip(state_np, action_np):
                next_s = self.transition(s, a)
                results.append(next_s)
            return jnp.array(results)
        else:
            next_s = self.transition(state_np, action_np)
            return jnp.array(next_s)
    
    def jax_model_transition(self, state, action):
        """JAX-compatible model transition."""
        if jnp is None:
            raise RuntimeError("jax_model_transition requires JAX to be installed.")
        
        state_np = np.asarray(state)
        action_np = np.asarray(action)
        
        if state_np.ndim > 1:
            results = []
            for s, a in zip(state_np, action_np):
                next_s = self.model_transition(s, a)
                results.append(next_s)
            return jnp.array(results)
        else:
            next_s = self.model_transition(state_np, action_np)
            return jnp.array(next_s)
    
    def jax_env_transition(self, state, action):
        """JAX-compatible environment transition."""
        return self.jax_transition(state, action)


# Register environment to registry
if REGISTRY_AVAILABLE and register_env is not None:
    register_env("drone_full_3d", DroneFull3DEnv)

