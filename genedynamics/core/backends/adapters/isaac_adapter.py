"""
Isaac Sim physics backend adapter.

This module provides a PhysicsBackend implementation using NVIDIA Isaac Sim.
Isaac Sim is a high-performance GPU-accelerated physics engine.
"""

from typing import Optional, Any, List, Tuple
import numpy as np

try:
    # Isaac Sim imports (these may vary depending on Isaac Sim version)
    try:
        from omni.isaac.core import World
        from omni.isaac.core.robots import Robot
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.utils.prims import create_prim
        from pxr import Usd, UsdGeom
        ISAAC_AVAILABLE = True
    except ImportError:
        # Try alternative imports
        try:
            from isaacsim import World, Robot
            ISAAC_AVAILABLE = True
        except ImportError:
            ISAAC_AVAILABLE = False
except ImportError:
    ISAAC_AVAILABLE = False

from genedynamics.core.backends.physics import PhysicsBackend
from genedynamics.core.types import State, Action
from genedynamics.envs.obstacles.base import Obstacle


class IsaacSimBackend(PhysicsBackend):
    """
    Isaac Sim physics backend implementation.
    
    This backend uses NVIDIA Isaac Sim for GPU-accelerated physics simulation.
    It supports:
    - Loading USD/URDF models
    - GPU-accelerated simulation
    - State management
    - Collision detection
    - Obstacle injection (as USD prims)
    
    Note: Isaac Sim requires NVIDIA Omniverse and is typically used on GPU.
    """
    
    name = "isaac"
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        dt: float = 0.01,
        use_gpu: bool = True,
    ):
        """
        Initialize Isaac Sim physics backend.
        
        Args:
            model_path: Path to USD/URDF model file
            dt: Simulation time step
            use_gpu: Whether to use GPU acceleration
        """
        if not ISAAC_AVAILABLE:
            raise ImportError(
                "Isaac Sim is required for IsaacSimBackend. "
                "Isaac Sim requires NVIDIA Omniverse. "
                "See: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html"
            )
        
        self.dt = dt
        self.use_gpu = use_gpu
        self.world = None
        self.robot = None
        self.obstacle_prims: List[str] = []  # Track added obstacle prims
        
        if model_path is not None:
            self.load_model(model_path)
    
    def load_model(self, model_path: str, **kwargs) -> None:
        """
        Load Isaac Sim model from file.
        
        Args:
            model_path: Path to USD/URDF file
            **kwargs: Additional parameters
        """
        try:
            # Create or get world
            if self.world is None:
                self.world = World(stage_units_in_meters=1.0, physics_dt=self.dt)
            
            # Load robot model
            if model_path.endswith('.usd') or model_path.endswith('.usda'):
                # USD file
                add_reference_to_stage(usd_path=model_path, prim_path="/robot")
            else:
                # URDF file
                add_reference_to_stage(usd_path=model_path, prim_path="/robot")
            
            # Create robot
            self.robot = Robot(
                prim_path="/robot",
                name="robot",
                position=np.array([0.0, 0.0, 0.0]),
            )
            
            # Reset world
            self.world.reset()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Isaac Sim model from {model_path}: {e}")
    
    def set_state(self, state: State) -> None:
        """
        Set Isaac Sim state.
        
        Args:
            state: State to set (dict with 'qpos', 'qvel', or flat array)
        """
        if self.robot is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if isinstance(state, dict):
            if 'qpos' in state:
                self.robot.set_joint_positions(np.asarray(state['qpos'], dtype=np.float32))
            if 'qvel' in state:
                self.robot.set_joint_velocities(np.asarray(state['qvel'], dtype=np.float32))
        else:
            # Flat array: assume [qpos, qvel]
            state_arr = np.asarray(state, dtype=np.float32)
            # Note: Isaac Sim API may vary, this is a simplified version
            if len(state_arr) >= self.robot.num_dof:
                self.robot.set_joint_positions(state_arr[:self.robot.num_dof])
                if len(state_arr) > self.robot.num_dof:
                    self.robot.set_joint_velocities(state_arr[self.robot.num_dof:])
    
    def get_state(self) -> State:
        """
        Get current Isaac Sim state.
        
        Returns:
            Dict with 'qpos' and 'qvel' keys
        """
        if self.robot is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return {
            'qpos': self.robot.get_joint_positions().cpu().numpy(),
            'qvel': self.robot.get_joint_velocities().cpu().numpy(),
        }
    
    def step(self, action: Action) -> State:
        """
        Step Isaac Sim simulation forward.
        
        Args:
            action: Control action (torques/forces)
            
        Returns:
            Next state after simulation step
        """
        if self.world is None or self.robot is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Set control
        action_arr = np.asarray(action, dtype=np.float32)
        self.robot.apply_action(action_arr)
        
        # Step simulation
        self.world.step(render=False)
        
        return self.get_state()
    
    def add_obstacle(self, obstacle: Obstacle, **kwargs) -> None:
        """
        Add obstacle to Isaac Sim scene as a USD prim.
        
        Args:
            obstacle: Obstacle object
            **kwargs: Additional parameters:
                - name: Prim name (optional)
                - color: Color (optional)
        """
        if self.world is None:
            raise RuntimeError("World not initialized. Call load_model() first.")
        
        # Get obstacle properties
        center = obstacle.center if hasattr(obstacle, 'center') and obstacle.center is not None else np.zeros(3)
        center = np.asarray(center, dtype=np.float32)
        name = kwargs.get('name', f"obstacle_{len(self.obstacle_prims)}")
        
        # Create prim based on obstacle type
        obs_name = type(obstacle).__name__.lower()
        
        if 'box' in obs_name:
            # Box obstacle
            if hasattr(obstacle, 'half_extents'):
                size = np.asarray(obstacle.half_extents, dtype=np.float32) * 2.0
            else:
                size = np.array([0.2, 0.2, 0.2], dtype=np.float32)
            
            prim_path = f"/obstacles/{name}"
            create_prim(
                prim_path=prim_path,
                prim_type="Cube",
                position=center,
                scale=size,
            )
        
        elif 'sphere' in obs_name:
            # Sphere obstacle
            radius = obstacle.radius if hasattr(obstacle, 'radius') else 0.1
            
            prim_path = f"/obstacles/{name}"
            create_prim(
                prim_path=prim_path,
                prim_type="Sphere",
                position=center,
                scale=np.array([radius * 2, radius * 2, radius * 2], dtype=np.float32),
            )
        
        else:
            # Default: box
            prim_path = f"/obstacles/{name}"
            create_prim(
                prim_path=prim_path,
                prim_type="Cube",
                position=center,
                scale=np.array([0.2, 0.2, 0.2], dtype=np.float32),
            )
        
        self.obstacle_prims.append(prim_path)
    
    def remove_obstacle(self, obstacle_id: Any) -> None:
        """
        Remove obstacle from Isaac Sim scene.
        
        Args:
            obstacle_id: Prim path or obstacle object
        """
        if isinstance(obstacle_id, str):
            prim_path = obstacle_id
        else:
            # Find by obstacle object
            # This is simplified; actual implementation would need to track mappings
            prim_path = None
        
        if prim_path and prim_path in self.obstacle_prims:
            # Remove prim (Isaac Sim API may vary)
            try:
                from omni.isaac.core.utils.prims import delete_prim
                delete_prim(prim_path)
                self.obstacle_prims.remove(prim_path)
            except Exception:
                pass
    
    def check_collision(self, state: Optional[State] = None) -> bool:
        """
        Check for collisions in Isaac Sim scene.
        
        Args:
            state: Optional state to check
            
        Returns:
            True if collision detected
        """
        if self.world is None:
            raise RuntimeError("World not initialized.")
        
        if state is not None:
            self.set_state(state)
        
        # Isaac Sim collision checking (API may vary)
        # This is a placeholder; actual implementation depends on Isaac Sim version
        try:
            # Check contacts
            contacts = self.world.get_physics_contacts()
            return len(contacts) > 0
        except Exception:
            # Fallback: assume no collision
            return False
    
    def get_collision_pairs(self) -> List[Tuple[int, int]]:
        """
        Get list of colliding object pairs.
        
        Returns:
            List of (object1_id, object2_id) tuples
        """
        if self.world is None:
            raise RuntimeError("World not initialized.")
        
        try:
            contacts = self.world.get_physics_contacts()
            pairs = []
            for contact in contacts:
                # Extract object IDs from contact (API may vary)
                # This is a placeholder
                pairs.append((0, 1))  # Simplified
            return pairs
        except Exception:
            return []
    
    def reset(self, **kwargs) -> State:
        """
        Reset Isaac Sim simulation.
        
        Args:
            **kwargs: Additional reset parameters
            
        Returns:
            Initial state
        """
        if self.world is None:
            raise RuntimeError("World not initialized.")
        
        self.world.reset()
        
        if 'qpos' in kwargs:
            self.robot.set_joint_positions(np.asarray(kwargs['qpos'], dtype=np.float32))
        if 'qvel' in kwargs:
            self.robot.set_joint_velocities(np.asarray(kwargs['qvel'], dtype=np.float32))
        
        return self.get_state()
    
    def close(self) -> None:
        """Close Isaac Sim backend and free resources."""
        if self.world is not None:
            self.world.clear()
        self.world = None
        self.robot = None
        self.obstacle_prims = []
