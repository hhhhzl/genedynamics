"""
Drone environment plugin.

This plugin provides integration with drone/quadrotor environments
for the experiment framework.
"""

from typing import Dict, Any
import numpy as np

from enerdynamics.envs.robots.drone import DroneModel
from ...framework.base import EnvironmentPlugin


class DroneEnvironmentPlugin(EnvironmentPlugin):
    """
    Plugin for drone/quadrotor environment.
    
    This environment represents a 3D quadrotor with 6 DOF (position + orientation).
    """
    
    @property
    def name(self) -> str:
        """Environment name identifier."""
        return "drone"
    
    def create_env(self, config: Dict[str, Any]) -> Any:
        """
        Create drone environment instance.
        
        Args:
            config: Environment configuration dictionary with keys:
                - dt: Time step (default: 0.1)
                - horizon: Planning horizon (default: 80)
                - mass: Drone mass (default: 0.5)
                - gravity: Gravity acceleration (default: 9.81)
                - Any other DroneModel parameters
                
        Returns:
            DroneModel instance configured as environment
        """
        # Create drone model with configuration
        # Note: DroneModel is currently a model, not a full environment
        # For now, we return the model. In the future, this may need to wrap
        # it in an environment class that provides reset, step, etc.
        
        drone_config = {
            'mass': config.get('mass', 0.5),
            'Ixx': config.get('Ixx', 0.0023),
            'Iyy': config.get('Iyy', 0.0023),
            'Izz': config.get('Izz', 0.0046),
            'arm_length': config.get('arm_length', 0.17),
            'gravity': config.get('gravity', 9.81),
        }
        
        drone = DroneModel(**drone_config)
        
        # Add environment-like attributes
        drone.dt = config.get('dt', 0.1)
        drone.horizon = config.get('horizon', 80)
        drone.target = np.array(config.get('target', [0.0, 0.0, 1.0]), dtype=np.float32)
        
        return drone
    
    def create_energy(self) -> Any:
        """
        Create energy functional for drone environment.
        
        Returns:
            Energy functional instance
        """
        # TODO: Implement drone-specific energy functional
        # For now, return None and let the framework handle it
        from enerdynamics.core.energy import LegacyEnergyFunctional, EnergyTerm
        import jax.numpy as jnp
        
        def task_energy(x, u, ctx):
            """Task energy: minimize distance to target and velocity."""
            pos = x[:3]
            vel = x[3:6]
            target = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
            pos_err = jnp.sum((pos - target) ** 2)
            vel_err = jnp.sum(vel ** 2)
            return pos_err + 0.1 * vel_err
        
        return LegacyEnergyFunctional({
            "task": EnergyTerm(task_energy, 1.0),
        })
    
    def get_state_dim(self) -> int:
        """
        Get state dimension.
        
        Returns:
            State dimension (12 for position, velocity, orientation, angular velocity)
        """
        return 12
    
    def extract_position(self, state: np.ndarray) -> np.ndarray:
        """
        Extract position coordinates from state.
        
        Args:
            state: Full state vector (x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz)
            
        Returns:
            Position vector (x, y, z)
        """
        return np.asarray(state, dtype=np.float32)[:3]


class DroneBox3DPlugin(EnvironmentPlugin):
    """
    Plugin for drone_box_3d environment.
    
    This environment represents a simplified 3D drone with double integrator dynamics.
    State: [x, y, z, vx, vy, vz] (6D)
    """
    
    @property
    def name(self) -> str:
        """Environment name identifier."""
        return "drone_box_3d"
    
    def create_env(self, config: Dict[str, Any]) -> Any:
        """
        Create drone_box_3d environment instance.
        
        Args:
            config: Environment configuration dictionary with keys:
                - dt: Time step (default: 0.1)
                - horizon: Planning horizon (default: 80)
                - p_max: Position bounds (default: 2.0)
                - v_max: Velocity bounds (default: 2.0)
                - control_limit: Control limit (default: 1.0)
                - target: Target position [x, y, z] (default: [0.0, 0.0, 1.0])
                - Any other DroneBox3DEnv parameters
                
        Returns:
            DroneBox3DEnv instance
        """
        from enerdynamics.envs.factories import make_env
        return make_env(self.name, **config)
    
    def create_energy(self) -> Any:
        """
        Create energy functional for drone_box_3d environment.
        
        Returns:
            LegacyEnergyFunctional instance
        """
        from enerdynamics.envs.factories import make_energy
        return make_energy(self.name)
    
    def get_state_dim(self) -> int:
        """
        Get state dimension.
        
        Returns:
            State dimension (6 for position and velocity: x, y, z, vx, vy, vz)
        """
        return 6
    
    def extract_position(self, state: np.ndarray) -> np.ndarray:
        """
        Extract position coordinates from state.
        
        Args:
            state: Full state vector (x, y, z, vx, vy, vz)
            
        Returns:
            Position vector (x, y, z)
        """
        return np.asarray(state, dtype=np.float32)[:3]

