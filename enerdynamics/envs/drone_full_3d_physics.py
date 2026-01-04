"""
Unified physics-enabled quadrotor environment interface.

This module provides a unified interface for quadrotor environments that supports
multiple physics backends (DroneModel, MuJoCo, Isaac Sim) and computational backends
(NumPy, JAX). It enables flexible configuration through a single config file while
maintaining high performance through JAX-native dynamics for planning.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

from enerdynamics.envs.robots.drone import DroneModel

# Import JAX dynamics (optional)
if JAX_AVAILABLE:
    try:
        from enerdynamics.envs.utils.jax_dynamics import (
            jax_quadrotor_step,
            jax_project_state,
            jax_quadrotor_step_batch,
            jax_project_state_batch,
        )
    except ImportError:
        # JAX dynamics not available
        jax_quadrotor_step = None
        jax_project_state = None
        jax_quadrotor_step_batch = None
        jax_project_state_batch = None
else:
    jax_quadrotor_step = None
    jax_project_state = None
    jax_quadrotor_step_batch = None
    jax_project_state_batch = None

# Register environment to registry
try:
    from enerdynamics.core.registry.environments import register_env
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    register_env = None

Array = np.ndarray


@dataclass
class DroneFull3DPhysicsEnv:
    """
    Unified quadrotor environment with support for multiple physics backends.
    
    This environment provides a unified interface that supports:
    - Multiple physics backends: DroneModel (NumPy), MuJoCo, Isaac Sim
    - Multiple computational backends: NumPy, JAX
    - Multiple renderers: Matplotlib, MuJoCo, Isaac Sim
    - JAX-native dynamics for high-performance planning
    
    State: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz] (12D)
    Action: [T1, T2, T3, T4] (4 motor thrusts, normalized 0-1)
    
    The environment can operate in three modes:
    1. Pure physics engine mode: Uses physics backend for both planning and validation
    2. Hybrid mode: Uses JAX dynamics for planning, physics engine for validation
    3. Pure JAX mode: Uses JAX dynamics for both (when physics_backend=None)
    """
    
    # Environment parameters
    dt: float = 0.1
    horizon: int = 80
    p_max: float = 2.0
    v_max: float = 2.0
    act_dim: int = 4  # 4 motor thrusts
    target: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    vel_weight: float = 0.1
    orientation_weight: float = 0.1
    control_limit: float = 1.0  # Max motor thrust (normalized)
    
    # Physics parameters
    mass: float = 0.5
    Ixx: float = 0.0023
    Iyy: float = 0.0023
    Izz: float = 0.0046
    arm_length: float = 0.17
    kf: float = 3.16e-10
    km: float = 7.94e-12
    gravity: float = 9.81
    
    # Backend configuration
    physics_backend: Optional[str] = None  # 'drone_model', 'mujoco', 'isaac', None
    use_physics_backend: bool = True
    use_jax_dynamics: bool = True  # Use JAX dynamics for planning
    jax_jit: bool = True  # Enable JIT compilation for JAX functions
    
    # Renderer configuration
    renderer: Optional[str] = None  # 'matplotlib', 'mujoco', 'isaac'
    
    # Model paths (optional, set by specific backends)
    model_path_mujoco: Optional[str] = None
    model_path_isaac: Optional[str] = None
    
    def __post_init__(self):
        """Initialize environment based on configuration."""
        # Compute inertia tensor
        self.I = np.array([self.Ixx, self.Iyy, self.Izz], dtype=np.float32)
        self.I_inv = 1.0 / self.I
        
        # Initialize physics backend if specified
        self._physics_backend_instance = None
        if self.use_physics_backend and self.physics_backend == 'drone_model':
            self._physics_backend_instance = DroneModel(
                mass=self.mass,
                Ixx=self.Ixx,
                Iyy=self.Iyy,
                Izz=self.Izz,
                arm_length=self.arm_length,
                kf=self.kf,
                km=self.km,
                gravity=self.gravity,
            )
        
        # Initialize renderer (lazy loading)
        self._renderer_instance = None
        
        # Pre-compile JAX functions if JIT is enabled (warm up JIT cache)
        if (self.use_jax_dynamics and JAX_AVAILABLE and self.jax_jit and 
            jax_quadrotor_step is not None):
            # Pre-compile with dummy inputs to warm up JIT cache
            # This reduces first-call latency during planning
            dummy_state = jnp.zeros(12, dtype=jnp.float32)
            dummy_action = jnp.zeros(4, dtype=jnp.float32)
            dummy_I = jnp.array(self.I, dtype=jnp.float32)
            
            # Warm up single-step function
            _ = jax_quadrotor_step(
                dummy_state, dummy_action, self.dt,
                self.mass, dummy_I, self.arm_length,
                self.kf, self.km, self.gravity
            )
            
            # Warm up batched function (small batch for warmup)
            if jax_quadrotor_step_batch is not None:
                dummy_states = jnp.zeros((4, 12), dtype=jnp.float32)
                dummy_actions = jnp.zeros((4, 4), dtype=jnp.float32)
                _ = jax_quadrotor_step_batch(
                    dummy_states, dummy_actions, self.dt,
                    self.mass, dummy_I, self.arm_length,
                    self.kf, self.km, self.gravity
                )
            
            # Warm up state projection
            if jax_project_state is not None:
                _ = jax_project_state(dummy_state, self.p_max, self.v_max)
            
            # Warm up batched state projection
            if jax_project_state_batch is not None:
                _ = jax_project_state_batch(dummy_states, self.p_max, self.v_max)
    
    def reset(self, rng: Optional["jax.Array"] = None) -> Tuple[Array, dict]:
        """
        Sample an initial state.
        
        Args:
            rng: Optional JAX random key for reproducibility
            
        Returns:
            Tuple of (initial_state, info_dict)
        """
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
            if not JAX_AVAILABLE:
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
        """
        Project state to valid bounds.
        
        Args:
            x_next: State vector to project
            
        Returns:
            Projected state
        """
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
        """
        Compute cost for a given state.
        
        Args:
            state: State vector
            
        Returns:
            Cost value
        """
        pos = state[0:3]
        vel = state[3:6]
        euler = state[6:9]
        target = np.asarray(self.target, dtype=np.float32)
        
        pos_err = np.sum((pos - target) ** 2)
        vel_err = self.vel_weight * np.sum(vel ** 2)
        orientation_err = self.orientation_weight * np.sum(euler ** 2)
        
        return float(pos_err + vel_err + orientation_err)
    
    def jax_cost(self, state):
        """
        JAX-compatible cost function with support for batched inputs.
        
        High-performance JIT-compiled cost computation with automatic batching support.
        
        Args:
            state: State vector or batch of states, shape (..., 12) or (batch, 12)
            
        Returns:
            Cost value(s), shape () for single state or (batch,) for batched
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("jax_cost requires JAX to be installed.")
        
        # JIT-compile cost function if not already compiled
        if not hasattr(self, '_jax_cost_jit'):
            @jax.jit
            def _cost_fn(s):
                target = jnp.asarray(self.target, dtype=jnp.float32)
                pos = s[..., 0:3]
                vel = s[..., 3:6]
                euler = s[..., 6:9]
                
                pos_err = jnp.sum((pos - target) ** 2, axis=-1)
                vel_err = self.vel_weight * jnp.sum(vel ** 2, axis=-1)
                orientation_err = self.orientation_weight * jnp.sum(euler ** 2, axis=-1)
                
                return pos_err + vel_err + orientation_err
            
            self._jax_cost_jit = _cost_fn
        
        return self._jax_cost_jit(state)
    
    def step(self, x_next: Array, u: Array, t: int, info):
        """
        Project the proposed state, compute task cost, and advance time.
        
        Args:
            x_next: Proposed next state
            u: Action taken
            t: Current time step
            info: Additional info dict
            
        Returns:
            Tuple of (projected_state, cost, done, info)
        """
        x_proj = self._project_state(x_next)
        cost = self.cost(x_proj)
        done = (t + 1) >= self.horizon
        info_n = {}
        return x_proj, cost, done, info_n
    
    def transition(self, state: Array, action: Array) -> Array:
        """
        Deterministic quadrotor transition using physics backend or NumPy.
        
        This method uses the physics backend (if available) or falls back to
        DroneModel for computation. For planning with JAX backend, use jax_transition().
        
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
        
        # Use physics backend if available
        if self._physics_backend_instance is not None:
            next_state = self._physics_backend_instance.step(state, motor_thrusts, dt=self.dt)
        else:
            # Fallback: compute using basic dynamics (should be overridden by subclasses)
            # This is a placeholder - subclasses should implement proper physics
            raise NotImplementedError(
                "transition() must be implemented by subclass or physics_backend must be set"
            )
        
        return self._project_state(next_state)
    
    def rollout_actions(self, state: Array, actions: Array) -> Array:
        """
        Roll out a sequence of control actions.
        
        Args:
            state: Initial state
            actions: Sequence of actions, shape (horizon, act_dim)
            
        Returns:
            Trajectory of states, shape (horizon+1, state_dim)
        """
        x = np.asarray(state, dtype=np.float32)
        traj = [x]
        for act in np.asarray(actions, dtype=np.float32):
            x = self.transition(x, act)
            traj.append(x)
        return np.stack(traj, axis=0)
    
    def model_transition(self, state: Array, action: Array) -> Array:
        """
        Model transition without state projection.
        
        Used by planners that want to simulate without bounds.
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            Next state (without projection)
        """
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        motor_thrusts = np.clip(action, 0.0, self.control_limit)
        
        if self._physics_backend_instance is not None:
            return self._physics_backend_instance.step(state, motor_thrusts, dt=self.dt)
        else:
            raise NotImplementedError(
                "model_transition() must be implemented by subclass or physics_backend must be set"
            )
    
    def jax_transition(self, state, action):
        """
        JAX-compatible transition using pure JAX dynamics.
        
        This method uses JAX-native quadrotor dynamics for high-performance planning.
        It supports JIT compilation and batched computation.
        
        Args:
            state: Current state (JAX array), shape (12,) or (batch, 12)
            action: Motor thrusts (JAX array), shape (4,) or (batch, 4)
            
        Returns:
            Next state (JAX array), shape (12,) or (batch, 12)
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("jax_transition requires JAX to be installed.")
        
        if not self.use_jax_dynamics or jax_quadrotor_step is None:
            # Fallback: use NumPy transition and convert
            state_np = np.asarray(state)
            action_np = np.asarray(action)
            if state_np.ndim > 1:
                results = []
                for s, a in zip(state_np, action_np):
                    next_s = self.transition(s, a)
                    results.append(next_s)
                return jnp.array(results) if JAX_AVAILABLE else np.array(results)
            else:
                next_s = self.transition(state_np, action_np)
                return jnp.array(next_s) if JAX_AVAILABLE else next_s
        
        # Use pure JAX dynamics
        state = jnp.asarray(state, dtype=jnp.float32)
        action = jnp.asarray(action, dtype=jnp.float32)
        
        # Clip motor thrusts
        motor_thrusts = jnp.clip(action, 0.0, self.control_limit)
        
        # Compute next state using JAX dynamics
        if state.ndim > 1:
            # Batched computation
            if jax_quadrotor_step_batch is not None:
                next_state = jax_quadrotor_step_batch(
                    state, motor_thrusts, self.dt,
                    self.mass, self.I, self.arm_length,
                    self.kf, self.km, self.gravity
                )
            else:
                # Fallback: process batch sequentially
                next_states = []
                for s, m in zip(state, motor_thrusts):
                    ns = jax_quadrotor_step(
                        s, m, self.dt,
                        self.mass, self.I, self.arm_length,
                        self.kf, self.km, self.gravity
                    )
                    next_states.append(ns)
                next_state = jnp.stack(next_states)
        else:
            # Single state
            next_state = jax_quadrotor_step(
                state, motor_thrusts, self.dt,
                self.mass, self.I, self.arm_length,
                self.kf, self.km, self.gravity
            )
        
        # Project state to bounds
        if state.ndim > 1:
            if jax_project_state_batch is not None:
                next_state = jax_project_state_batch(next_state, self.p_max, self.v_max)
            else:
                # Fallback: process batch sequentially
                projected = []
                for ns in next_state:
                    pns = jax_project_state(ns, self.p_max, self.v_max)
                    projected.append(pns)
                next_state = jnp.stack(projected)
        else:
            next_state = jax_project_state(next_state, self.p_max, self.v_max)
        
        return next_state
    
    def jax_model_transition(self, state, action):
        """
        JAX-compatible model transition without state projection.
        
        Args:
            state: Current state (JAX array)
            action: Action to take (JAX array)
            
        Returns:
            Next state without projection (JAX array)
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("jax_model_transition requires JAX to be installed.")
        
        if not self.use_jax_dynamics or jax_quadrotor_step is None:
            # Fallback
            state_np = np.asarray(state)
            action_np = np.asarray(action)
            if state_np.ndim > 1:
                results = []
                for s, a in zip(state_np, action_np):
                    next_s = self.model_transition(s, a)
                    results.append(next_s)
                return jnp.array(results) if JAX_AVAILABLE else np.array(results)
            else:
                next_s = self.model_transition(state_np, action_np)
                return jnp.array(next_s) if JAX_AVAILABLE else next_s
        
        # Use pure JAX dynamics without projection
        state = jnp.asarray(state, dtype=jnp.float32)
        action = jnp.asarray(action, dtype=jnp.float32)
        motor_thrusts = jnp.clip(action, 0.0, self.control_limit)
        
        if state.ndim > 1:
            if jax_quadrotor_step_batch is not None:
                next_state = jax_quadrotor_step_batch(
                    state, motor_thrusts, self.dt,
                    self.mass, self.I, self.arm_length,
                    self.kf, self.km, self.gravity
                )
            else:
                # Fallback: process batch sequentially
                next_states = []
                for s, m in zip(state, motor_thrusts):
                    ns = jax_quadrotor_step(
                        s, m, self.dt,
                        self.mass, self.I, self.arm_length,
                        self.kf, self.km, self.gravity
                    )
                    next_states.append(ns)
                next_state = jnp.stack(next_states)
        else:
            next_state = jax_quadrotor_step(
                state, motor_thrusts, self.dt,
                self.mass, self.I, self.arm_length,
                self.kf, self.km, self.gravity
            )
        
        return next_state
    
    def jax_env_transition(self, state, action):
        """
        JAX-compatible environment transition (same as jax_transition).
        
        Args:
            state: Current state (JAX array)
            action: Action to take (JAX array)
            
        Returns:
            Next state with projection (JAX array)
        """
        return self.jax_transition(state, action)
    
    def render(self, state: Optional[Array] = None, mode: str = "human", **kwargs):
        """
        Render the environment using the configured renderer.
        
        High-performance rendering with support for multiple backends:
        - matplotlib: 2D/3D Matplotlib visualization
        - mujoco: MuJoCo interactive viewer and RGB rendering
        - isaac: Isaac Sim GPU-accelerated rendering
        
        Args:
            state: Optional state to render (if None, uses current state)
            mode: Rendering mode ('human', 'rgb_array', 'depth', 'rgbd')
            **kwargs: Additional rendering parameters
            
        Returns:
            Rendered output (depends on mode and renderer)
        """
        if self.renderer == "matplotlib":
            # Matplotlib rendering (handled by visualization plugins)
            return None
        
        elif self.renderer == "mujoco":
            # MuJoCo rendering
            if self._mujoco_backend is not None:
                from enerdynamics.core.backends.render_impl import MujocoRenderer
                if self._renderer_instance is None:
                    self._renderer_instance = MujocoRenderer(
                        mujoco_model=self._mujoco_backend.model,
                        mujoco_data=self._mujoco_backend.data
                    )
                
                # Update state if provided
                if state is not None:
                    from enerdynamics.envs.utils.state_converter import state_12d_to_mujoco
                    mujoco_state = state_12d_to_mujoco(state)
                    self._mujoco_backend.set_state(mujoco_state)
                
                return self._renderer_instance.render(state=None, mode=mode, **kwargs)
            return None
        
        elif self.renderer == "isaac":
            # Isaac Sim rendering
            if self._isaac_backend is not None:
                from enerdynamics.core.backends.render_impl import IsaacSimRenderer
                if self._renderer_instance is None:
                    world = self._isaac_backend.world if hasattr(self._isaac_backend, 'world') else None
                    self._renderer_instance = IsaacSimRenderer(world=world)
                
                # Update state if provided
                if state is not None:
                    from enerdynamics.envs.utils.state_converter import state_12d_to_isaac
                    isaac_state = state_12d_to_isaac(state)
                    self._isaac_backend.set_state(isaac_state)
                
                return self._renderer_instance.render(state=None, mode=mode, **kwargs)
            return None
        
        # Default: no rendering
        return None
    
    def close(self):
        """Close environment and free resources."""
        if self._physics_backend_instance is not None:
            if hasattr(self._physics_backend_instance, 'close'):
                self._physics_backend_instance.close()
        
        if self._renderer_instance is not None:
            if hasattr(self._renderer_instance, 'close'):
                self._renderer_instance.close()


# Register environment to registry
if REGISTRY_AVAILABLE and register_env is not None:
    register_env("drone_full_3d_physics", DroneFull3DPhysicsEnv)

