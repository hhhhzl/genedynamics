"""
MuJoCo adapter for full 3D quadrotor environment.

This module provides a MuJoCo-based implementation of the unified quadrotor
environment. It uses MuJoCo for physics simulation while maintaining compatibility
with the unified interface and JAX dynamics for planning.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import tempfile
import os

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    mujoco = None

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

from genedynamics.envs.drone_full_3d_physics import DroneFull3DPhysicsEnv
from genedynamics.core.backends.adapters.mujoco_adapter import MujocoPhysicsBackend
from genedynamics.envs.utils.state_converter import (
    state_12d_to_mujoco,
    mujoco_to_state_12d,
)
from genedynamics.envs.utils.mujoco_model_generator import (
    generate_mujoco_xml_with_obstacles,
    create_base_quadrotor_xml,
)
from genedynamics.envs.obstacles.base import ObstacleManager

Array = np.ndarray


@dataclass
class DroneFull3DMujocoEnv(DroneFull3DPhysicsEnv):
    """
    Full 3D quadrotor environment with MuJoCo physics backend.
    
    This environment uses MuJoCo for high-fidelity physics simulation while
    maintaining compatibility with the unified interface. It supports:
    - MuJoCo physics simulation for validation
    - JAX dynamics for high-performance planning
    - Obstacle integration via XML generation
    - MuJoCo viewer for visualization
    
    State: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz] (12D)
    Action: [T1, T2, T3, T4] (4 motor thrusts, normalized 0-1)
    """
    
    model_path: Optional[str] = None
    use_mujoco_physics: bool = True
    obstacles: Optional[ObstacleManager] = None
    
    def __post_init__(self):
        """Initialize MuJoCo backend with optional obstacles."""
        # Initialize base class
        super().__post_init__()
        
        # Override physics backend setting
        self.physics_backend = 'mujoco'
        self.use_physics_backend = self.use_mujoco_physics
        
        # Initialize MuJoCo backend
        if self.use_mujoco_physics and MUJOCO_AVAILABLE:
            # Handle obstacles: MuJoCo requires obstacles in XML
            if self.obstacles is not None and len(self.obstacles) > 0:
                # Generate XML with obstacles
                if self.model_path is None:
                    # Create base model if no path provided
                    temp_base = tempfile.NamedTemporaryFile(
                        mode='w', suffix='.xml', delete=False
                    )
                    create_base_quadrotor_xml(
                        temp_base.name,
                        mass=self.mass,
                        arm_length=self.arm_length
                    )
                    base_path = temp_base.name
                    temp_base.close()
                else:
                    base_path = self.model_path
                
                # Generate XML with obstacles
                temp_xml = tempfile.NamedTemporaryFile(
                    mode='w', suffix='.xml', delete=False
                )
                temp_xml.close()
                
                generate_mujoco_xml_with_obstacles(
                    base_path,
                    self.obstacles,
                    output_path=temp_xml.name
                )
                
                actual_model_path = temp_xml.name
                self._temp_model_path = temp_xml.name
                
                # Clean up base temp file if we created it
                if self.model_path is None:
                    os.unlink(base_path)
            else:
                # No obstacles, use provided or create base model
                if self.model_path is None:
                    temp_xml = tempfile.NamedTemporaryFile(
                        mode='w', suffix='.xml', delete=False
                    )
                    temp_xml.close()
                    create_base_quadrotor_xml(
                        temp_xml.name,
                        mass=self.mass,
                        arm_length=self.arm_length
                    )
                    actual_model_path = temp_xml.name
                    self._temp_model_path = temp_xml.name
                else:
                    actual_model_path = self.model_path
                    self._temp_model_path = None
            
            # Initialize MuJoCo backend
            self._physics_backend_instance = MujocoPhysicsBackend(
                model_path=actual_model_path,
                dt=self.dt,
            )
        else:
            self._physics_backend_instance = None
            self._temp_model_path = None
    
    def _state_to_mujoco(self, state: Array) -> Dict[str, np.ndarray]:
        """
        Convert 12D state to MuJoCo state format.
        
        Args:
            state: 12D state [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
            
        Returns:
            MuJoCo state dict with 'qpos' and 'qvel'
        """
        return state_12d_to_mujoco(state)
    
    def _mujoco_to_state(self, mujoco_state: Dict[str, np.ndarray]) -> Array:
        """
        Convert MuJoCo state to 12D state.
        
        Args:
            mujoco_state: MuJoCo state dict with 'qpos' and 'qvel'
            
        Returns:
            12D state array
        """
        return mujoco_to_state_12d(mujoco_state)
    
    def transition(self, state: Array, action: Array) -> Array:
        """
        Deterministic quadrotor transition using MuJoCo physics.
        
        This method uses MuJoCo for physics simulation. For planning,
        use jax_transition() which uses JAX dynamics for better performance.
        
        Args:
            state: Current state [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
            action: Motor thrusts [T1, T2, T3, T4] (normalized 0-1)
            
        Returns:
            Next state
        """
        if not self.use_mujoco_physics or self._physics_backend_instance is None:
            # Fallback to base class (DroneModel)
            return super().transition(state, action)
        
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        
        # Clip motor thrusts
        motor_thrusts = np.clip(action, 0.0, self.control_limit)
        
        # Convert state to MuJoCo format
        mujoco_state = self._state_to_mujoco(state)
        self._physics_backend_instance.set_state(mujoco_state)
        
        # Step MuJoCo simulation
        # Map motor thrusts to MuJoCo control (assuming 4 actuators for 4 motors)
        # Motor thrusts are normalized [0, 1], map to control range
        mujoco_ctrl = motor_thrusts.astype(np.float64)
        next_mujoco_state = self._physics_backend_instance.step(mujoco_ctrl)
        
        # Convert back to 12D state
        next_state = self._mujoco_to_state(next_mujoco_state)
        
        return self._project_state(next_state)
    
    def model_transition(self, state: Array, action: Array) -> Array:
        """
        Model transition without state projection using MuJoCo.
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            Next state without projection
        """
        if not self.use_mujoco_physics or self._physics_backend_instance is None:
            return super().model_transition(state, action)
        
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        motor_thrusts = np.clip(action, 0.0, self.control_limit)
        
        mujoco_state = self._state_to_mujoco(state)
        self._physics_backend_instance.set_state(mujoco_state)
        next_mujoco_state = self._physics_backend_instance.step(motor_thrusts)
        
        return self._mujoco_to_state(next_mujoco_state)
    
    def render(self, state: Optional[Array] = None, mode: str = "human", **kwargs):
        """
        Render using MuJoCo renderer.
        
        High-performance rendering with support for interactive viewer and RGB array.
        
        Args:
            state: Optional state to render (if None, uses current state)
            mode: Rendering mode:
                - 'human': Interactive MuJoCo viewer (blocking)
                - 'rgb_array': Return RGB image array
                - 'depth': Return depth image array
            **kwargs: Additional rendering parameters (width, height, etc.)
            
        Returns:
            Rendered output (depends on mode)
        """
        if not MUJOCO_AVAILABLE or self._physics_backend_instance is None:
            return None
        
        # Use parent class renderer if renderer is set to 'mujoco'
        if self.renderer == "mujoco":
            return super().render(state, mode, **kwargs)
        
        # Fallback: direct MuJoCo viewer
        if mode == "human":
            if state is not None:
                mujoco_state = self._state_to_mujoco(state)
                self._physics_backend_instance.set_state(mujoco_state)
            
            # Launch MuJoCo viewer
            model = self._physics_backend_instance.model
            data = self._physics_backend_instance.data
            
            if model is not None and data is not None:
                with mujoco.viewer.launch_passive(model, data) as viewer:
                    # Sync viewer with current state
                    viewer.sync()
                    # Keep viewer open (user closes manually)
                    input("Press Enter to close MuJoCo viewer...")
        
        return None
    
    def close(self):
        """Close MuJoCo backend and cleanup temporary files."""
        if self._physics_backend_instance is not None:
            self._physics_backend_instance.close()
        
        # Cleanup temporary model file
        if self._temp_model_path and os.path.exists(self._temp_model_path):
            os.unlink(self._temp_model_path)
            self._temp_model_path = None
        
        super().close()

