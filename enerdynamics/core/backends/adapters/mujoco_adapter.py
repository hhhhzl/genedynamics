"""
MuJoCo physics backend adapter.

This module provides a PhysicsBackend implementation using MuJoCo.
MuJoCo is a high-performance physics engine for robotics simulation.
"""

from typing import Optional, Any, List, Tuple
import numpy as np

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    mujoco = None
    # Skip defining MujocoPhysicsBackend so adapters/backends can optional-import cleanly
    raise ImportError("mujoco is not installed. Install with: pip install mujoco") from None

from enerdynamics.core.backends.physics import PhysicsBackend
from enerdynamics.core.types import State, Action
from enerdynamics.envs.obstacles.base import Obstacle


class MujocoPhysicsBackend(PhysicsBackend):
    """
    MuJoCo physics backend implementation.
    
    This backend uses MuJoCo for high-performance physics simulation.
    It supports:
    - Loading MJCF/URDF models
    - State management
    - Dynamics computation
    - Collision detection
    - Obstacle injection (as MuJoCo geoms)
    """
    
    name = "mujoco"
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_xml: Optional[str] = None,
        dt: float = 0.01,
    ):
        """
        Initialize MuJoCo physics backend.
        
        Args:
            model_path: Path to MJCF/URDF model file
            model_xml: XML string of model (alternative to model_path)
            dt: Simulation time step
        """
        if not MUJOCO_AVAILABLE:
            raise ImportError(
                "MuJoCo is required for MujocoPhysicsBackend. "
                "Install with: pip install mujoco"
            )
        
        self.dt = dt
        self.model = None
        self.data = None
        self.obstacle_geom_ids: List[int] = []  # Track added obstacle geoms
        
        if model_path is not None:
            self.load_model(model_path)
        elif model_xml is not None:
            self.load_model_from_xml(model_xml)
    
    def load_model(self, model_path: str, **kwargs) -> None:
        """
        Load MuJoCo model from file.
        
        Args:
            model_path: Path to MJCF/URDF file
            **kwargs: Additional parameters (ignored for now)
        """
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
            # Set time step
            self.model.opt.timestep = self.dt
        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model from {model_path}: {e}")
    
    def load_model_from_xml(self, model_xml: str) -> None:
        """
        Load MuJoCo model from XML string.
        
        Args:
            model_xml: XML string of model
        """
        try:
            self.model = mujoco.MjModel.from_xml_string(model_xml)
            self.data = mujoco.MjData(self.model)
            self.model.opt.timestep = self.dt
        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model from XML: {e}")
    
    def set_state(self, state: State) -> None:
        """
        Set MuJoCo state.
        
        Args:
            state: State to set (can be dict with 'qpos', 'qvel', or flat array)
        """
        if self.data is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if isinstance(state, dict):
            # Dict format: {'qpos': ..., 'qvel': ...}
            if 'qpos' in state:
                self.data.qpos[:] = np.asarray(state['qpos'], dtype=np.float64)
            if 'qvel' in state:
                self.data.qvel[:] = np.asarray(state['qvel'], dtype=np.float64)
        else:
            # Flat array: assume [qpos, qvel]
            state_arr = np.asarray(state, dtype=np.float64)
            nq = self.model.nq
            nv = self.model.nv
            
            if len(state_arr) == nq + nv:
                self.data.qpos[:] = state_arr[:nq]
                self.data.qvel[:] = state_arr[nq:nq+nv]
            elif len(state_arr) == nq:
                self.data.qpos[:] = state_arr
                self.data.qvel[:] = 0.0
            else:
                raise ValueError(
                    f"State dimension mismatch: expected {nq + nv} or {nq}, "
                    f"got {len(state_arr)}"
                )
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
    
    def get_state(self) -> State:
        """
        Get current MuJoCo state.
        
        Returns:
            Dict with 'qpos' and 'qvel' keys
        """
        if self.data is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return {
            'qpos': self.data.qpos.copy(),
            'qvel': self.data.qvel.copy(),
        }
    
    def step(self, action: Action) -> State:
        """
        Step MuJoCo simulation forward.
        
        Args:
            action: Control action (torques/forces)
            
        Returns:
            Next state after simulation step
        """
        if self.data is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Set control
        action_arr = np.asarray(action, dtype=np.float64)
        if len(action_arr) <= self.model.nu:
            self.data.ctrl[:len(action_arr)] = action_arr
        else:
            raise ValueError(
                f"Action dimension mismatch: expected <= {self.model.nu}, "
                f"got {len(action_arr)}"
            )
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        return self.get_state()
    
    def add_obstacle(self, obstacle: Obstacle, **kwargs) -> None:
        """
        Add obstacle to MuJoCo scene as a geom.
        
        Args:
            obstacle: Obstacle object (should implement Obstacle protocol)
            **kwargs: Additional parameters:
                - name: Geom name (optional)
                - rgba: Color (optional, default [0.8, 0.2, 0.2, 0.5])
                - group: Geom group (optional, default 0)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Convert obstacle to MuJoCo geom
        geom_xml = self._obstacle_to_geom_xml(obstacle, **kwargs)
        
        # Note: MuJoCo doesn't support dynamic model modification at runtime
        # We would need to rebuild the model. For now, we track obstacles
        # and they can be added when the model is loaded.
        # This is a limitation of MuJoCo's design.
        
        # Store obstacle info for later use
        if not hasattr(self, '_obstacle_geoms'):
            self._obstacle_geoms = []
        
        self._obstacle_geoms.append({
            'obstacle': obstacle,
            'geom_xml': geom_xml,
            'kwargs': kwargs,
        })
    
    def _obstacle_to_geom_xml(self, obstacle: Obstacle, **kwargs) -> str:
        """
        Convert obstacle to MuJoCo geom XML.
        
        Args:
            obstacle: Obstacle object
            **kwargs: Additional parameters
            
        Returns:
            XML string for MuJoCo geom
        """
        name = kwargs.get('name', f"obstacle_{len(self.obstacle_geom_ids)}")
        rgba = kwargs.get('rgba', [0.8, 0.2, 0.2, 0.5])
        group = kwargs.get('group', 0)
        
        # Get obstacle properties
        center = obstacle.center if hasattr(obstacle, 'center') and obstacle.center is not None else np.zeros(3)
        center = np.asarray(center, dtype=np.float64)
        
        # Convert based on obstacle type
        if hasattr(obstacle, 'name'):
            obs_name = obstacle.name.lower()
        else:
            obs_name = type(obstacle).__name__.lower()
        
        if 'box' in obs_name or 'BoxObstacle' in type(obstacle).__name__:
            # Box obstacle
            if hasattr(obstacle, 'half_extents'):
                half_extents = np.asarray(obstacle.half_extents, dtype=np.float64)
                size_str = ' '.join(map(str, half_extents))
            else:
                size_str = "0.1 0.1 0.1"
            
            xml = f"""
            <geom name="{name}" type="box" pos="{center[0]} {center[1]} {center[2]}" 
                  size="{size_str}" rgba="{' '.join(map(str, rgba))}" group="{group}"/>
            """
        
        elif 'sphere' in obs_name or 'SphereObstacle' in type(obstacle).__name__:
            # Sphere obstacle
            radius = obstacle.radius if hasattr(obstacle, 'radius') else 0.1
            
            xml = f"""
            <geom name="{name}" type="sphere" pos="{center[0]} {center[1]} {center[2]}" 
                  size="{radius}" rgba="{' '.join(map(str, rgba))}" group="{group}"/>
            """
        
        elif 'cylinder' in obs_name or 'CylinderObstacle' in type(obstacle).__name__:
            # Cylinder obstacle
            radius = obstacle.radius if hasattr(obstacle, 'radius') else 0.1
            height = obstacle.height if hasattr(obstacle, 'height') else 0.2
            
            xml = f"""
            <geom name="{name}" type="cylinder" pos="{center[0]} {center[1]} {center[2]}" 
                  size="{radius} {height/2}" rgba="{' '.join(map(str, rgba))}" group="{group}"/>
            """
        
        else:
            # Default: box approximation
            xml = f"""
            <geom name="{name}" type="box" pos="{center[0]} {center[1]} {center[2]}" 
                  size="0.1 0.1 0.1" rgba="{' '.join(map(str, rgba))}" group="{group}"/>
            """
        
        return xml.strip()
    
    def remove_obstacle(self, obstacle_id: Any) -> None:
        """
        Remove obstacle from MuJoCo scene.
        
        Note: MuJoCo doesn't support dynamic model modification.
        This method is a placeholder for API compatibility.
        
        Args:
            obstacle_id: Obstacle identifier
        """
        # MuJoCo limitation: cannot remove geoms at runtime
        # Would need to rebuild model
        if hasattr(self, '_obstacle_geoms'):
            # Remove from tracking
            self._obstacle_geoms = [
                g for g in self._obstacle_geoms
                if g['obstacle'] != obstacle_id
            ]
    
    def check_collision(self, state: Optional[State] = None) -> bool:
        """
        Check for collisions in MuJoCo scene.
        
        Args:
            state: Optional state to check (if None, uses current state)
            
        Returns:
            True if collision detected, False otherwise
        """
        if self.data is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Set state if provided
        if state is not None:
            self.set_state(state)
        
        # Check contacts
        ncon = self.data.ncon
        return ncon > 0
    
    def get_collision_pairs(self) -> List[Tuple[int, int]]:
        """
        Get list of colliding geom pairs.
        
        Returns:
            List of (geom1_id, geom2_id) tuples
        """
        if self.data is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        pairs = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            pairs.append((int(geom1), int(geom2)))
        
        return pairs
    
    def reset(self, **kwargs) -> State:
        """
        Reset MuJoCo simulation to initial state.
        
        Args:
            **kwargs: Additional reset parameters:
                - qpos: Initial joint positions (optional)
                - qvel: Initial joint velocities (optional)
                
        Returns:
            Initial state
        """
        if self.data is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Reset to default
        mujoco.mj_resetData(self.model, self.data)
        
        # Override with kwargs if provided
        if 'qpos' in kwargs:
            self.data.qpos[:] = np.asarray(kwargs['qpos'], dtype=np.float64)
        if 'qvel' in kwargs:
            self.data.qvel[:] = np.asarray(kwargs['qvel'], dtype=np.float64)
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        return self.get_state()
    
    def close(self) -> None:
        """Close MuJoCo backend and free resources."""
        self.model = None
        self.data = None
        self.obstacle_geom_ids = []
        if hasattr(self, '_obstacle_geoms'):
            self._obstacle_geoms = []
