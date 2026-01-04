"""
Isaac Sim adapter for full 3D quadrotor environment.

This module provides an Isaac Sim-based implementation of the unified quadrotor
environment. It uses Isaac Sim for GPU-accelerated physics simulation while
maintaining compatibility with the unified interface and JAX dynamics for planning.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import os

try:
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.prims import create_prim
    from omni.isaac.core.utils.types import ArticulationAction
    import torch
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False
    World = None
    Robot = None
    torch = None

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

from enerdynamics.envs.drone_full_3d_physics import DroneFull3DPhysicsEnv
from enerdynamics.core.backends.adapters.isaac_adapter import IsaacSimBackend
from enerdynamics.envs.utils.state_converter import (
    state_12d_to_isaac,
    isaac_to_state_12d,
)
from enerdynamics.envs.utils.isaac_usd_generator import (
    create_quadrotor_usd_with_obstacles,
    create_base_quadrotor_usd,
)
from enerdynamics.envs.obstacles.base import ObstacleManager

Array = np.ndarray


@dataclass
class DroneFull3DIsaacEnv(DroneFull3DPhysicsEnv):
    """
    Full 3D quadrotor environment with Isaac Sim physics backend.
    
    This environment uses Isaac Sim for GPU-accelerated physics simulation while
    maintaining compatibility with the unified interface. It supports:
    - Isaac Sim physics simulation for validation
    - JAX dynamics for high-performance planning
    - Runtime obstacle addition (Isaac Sim supports dynamic obstacles)
    - Isaac Sim rendering for visualization
    
    State: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz] (12D)
    Action: [T1, T2, T3, T4] (4 motor thrusts, normalized 0-1)
    """
    
    model_path: Optional[str] = None
    use_isaac_physics: bool = True
    use_gpu: bool = True
    device: str = "cuda"  # or "cpu"
    obstacles: Optional[ObstacleManager] = None
    
    def __post_init__(self):
        """Initialize Isaac Sim backend with optional obstacles."""
        # Initialize base class
        super().__post_init__()
        
        # Override physics backend setting
        self.physics_backend = 'isaac'
        self.use_physics_backend = self.use_isaac_physics
        
        # Initialize Isaac Sim backend
        if self.use_isaac_physics and ISAAC_AVAILABLE:
            # Handle obstacles: Isaac Sim supports runtime addition, but we can pre-define
            if self.model_path is None:
                # Create base USD model
                import tempfile
                temp_usd = tempfile.NamedTemporaryFile(
                    mode='w', suffix='.usd', delete=False
                )
                temp_usd.close()
                
                create_quadrotor_usd_with_obstacles(
                    temp_usd.name,
                    obstacles=self.obstacles,
                    mass=self.mass,
                    arm_length=self.arm_length
                )
                actual_model_path = temp_usd.name
                self._temp_model_path = temp_usd.name
            else:
                actual_model_path = self.model_path
                self._temp_model_path = None
            
            # Initialize Isaac Sim backend
            self._physics_backend_instance = IsaacSimBackend(
                model_path=actual_model_path,
                dt=self.dt,
                use_gpu=self.use_gpu,
            )
            
            # Add obstacles at runtime if not in USD file
            if self.obstacles is not None and len(self.obstacles) > 0:
                for obstacle in self.obstacles:
                    self._physics_backend_instance.add_obstacle(obstacle)
            
            # Device for tensor operations
            if torch is not None:
                self.device = torch.device(
                    self.device if self.use_gpu and torch.cuda.is_available() else "cpu"
                )
        else:
            self._physics_backend_instance = None
            self._temp_model_path = None
            self.device = None
    
    def _state_to_isaac(self, state: Array) -> Dict[str, np.ndarray]:
        """
        Convert 12D state to Isaac Sim state format.
        
        Args:
            state: 12D state [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
            
        Returns:
            Isaac Sim state dict with 'qpos' and 'qvel'
        """
        return state_12d_to_isaac(state)
    
    def _isaac_to_state(self, isaac_state: Dict[str, Any]) -> Array:
        """
        Convert Isaac Sim state to 12D state.
        
        Handles both NumPy arrays and PyTorch tensors.
        
        Args:
            isaac_state: Isaac Sim state dict with 'qpos' and 'qvel'
            
        Returns:
            12D state array
        """
        return isaac_to_state_12d(isaac_state)
    
    def transition(self, state: Array, action: Array) -> Array:
        """
        Deterministic quadrotor transition using Isaac Sim physics.
        
        This method uses Isaac Sim for physics simulation. For planning,
        use jax_transition() which uses JAX dynamics for better performance.
        
        Args:
            state: Current state [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
            action: Motor thrusts [T1, T2, T3, T4] (normalized 0-1)
            
        Returns:
            Next state
        """
        if not self.use_isaac_physics or self._physics_backend_instance is None:
            # Fallback to base class (DroneModel)
            return super().transition(state, action)
        
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        
        # Clip motor thrusts
        motor_thrusts = np.clip(action, 0.0, self.control_limit)
        
        # Convert state to Isaac Sim format
        isaac_state = self._state_to_isaac(state)
        self._physics_backend_instance.set_state(isaac_state)
        
        # Step Isaac Sim simulation
        # Convert motor thrusts to ArticulationAction
        if ISAAC_AVAILABLE and ArticulationAction is not None:
            articulation_action = ArticulationAction(
                joint_efforts=motor_thrusts,  # Direct motor thrusts as efforts
            )
            next_isaac_state = self._physics_backend_instance.step(articulation_action)
        else:
            next_isaac_state = self._physics_backend_instance.step(motor_thrusts)
        
        # Convert back to 12D state
        next_state = self._isaac_to_state(next_isaac_state)
        
        return self._project_state(next_state)
    
    def model_transition(self, state: Array, action: Array) -> Array:
        """
        Model transition without state projection using Isaac Sim.
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            Next state without projection
        """
        if not self.use_isaac_physics or self._physics_backend_instance is None:
            return super().model_transition(state, action)
        
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        motor_thrusts = np.clip(action, 0.0, self.control_limit)
        
        isaac_state = self._state_to_isaac(state)
        self._physics_backend_instance.set_state(isaac_state)
        
        if ISAAC_AVAILABLE and ArticulationAction is not None:
            articulation_action = ArticulationAction(joint_efforts=motor_thrusts)
            next_isaac_state = self._physics_backend_instance.step(articulation_action)
        else:
            next_isaac_state = self._physics_backend_instance.step(motor_thrusts)
        
        return self._isaac_to_state(next_isaac_state)
    
    def add_obstacles_to_physics(self, obstacles: ObstacleManager):
        """
        Add obstacles to Isaac Sim scene at runtime.
        
        Isaac Sim supports runtime obstacle addition, unlike MuJoCo.
        
        Args:
            obstacles: ObstacleManager with obstacles to add
        """
        if self._physics_backend_instance is not None:
            for obstacle in obstacles:
                self._physics_backend_instance.add_obstacle(obstacle)
    
    def render(self, state: Optional[Array] = None, mode: str = "human", **kwargs):
        """
        Render using Isaac Sim renderer.
        
        GPU-accelerated rendering with support for interactive viewport and RGB array.
        
        Args:
            state: Optional state to render (if None, uses current state)
            mode: Rendering mode:
                - 'human': Interactive Isaac Sim viewport
                - 'rgb_array': Return RGB image array (GPU-accelerated)
                - 'depth': Return depth image array
                - 'rgbd': Return RGB + depth dictionary
            **kwargs: Additional rendering parameters (camera_name, width, height, etc.)
            
        Returns:
            Rendered output (depends on mode)
        """
        if not ISAAC_AVAILABLE or self._physics_backend_instance is None:
            return None
        
        # Use parent class renderer if renderer is set to 'isaac'
        if self.renderer == "isaac":
            return super().render(state, mode, **kwargs)
        
        # Fallback: direct Isaac Sim rendering
        if state is not None:
            isaac_state = self._state_to_isaac(state)
            self._physics_backend_instance.set_state(isaac_state)
        
        # Isaac Sim viewer is typically managed by the World instance
        if mode == "human" and self._physics_backend_instance.world is not None:
            # Step with rendering enabled
            self._physics_backend_instance.world.step(render=True)
        
        return None
    
    def close(self):
        """Close Isaac Sim backend and cleanup temporary files."""
        if self._physics_backend_instance is not None:
            self._physics_backend_instance.close()
        
        # Cleanup temporary model file
        if self._temp_model_path and os.path.exists(self._temp_model_path):
            os.unlink(self._temp_model_path)
            self._temp_model_path = None
        
        super().close()

