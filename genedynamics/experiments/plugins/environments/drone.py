"""
Drone environment plugin.

This plugin provides integration with drone/quadrotor environments
for the experiment framework.
"""

from typing import Dict, Any
import numpy as np

from genedynamics.envs.robots.drone import DroneModel
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
        from genedynamics.core.energy import LegacyEnergyFunctional, EnergyTerm
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

    def get_position_dim(self) -> int:
        """Position dimension (3 for 3D)."""
        return 3

    def get_position_dim(self) -> int:
        """Position dimension (3 for 3D)."""
        return 3


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
        from genedynamics.envs.factories import make_env
        return make_env(self.name, **config)
    
    def create_energy(self) -> Any:
        """
        Create energy functional for drone_box_3d environment.
        
        Returns:
            LegacyEnergyFunctional instance
        """
        from genedynamics.envs.factories import make_energy
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

    def get_position_dim(self) -> int:
        """Position dimension (3 for 3D)."""
        return 3


class DroneFull3DPlugin(EnvironmentPlugin):
    """
    Plugin for drone_full_3d environment.
    
    This environment represents a full 3D quadrotor with complete dynamics.
    State: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz] (12D)
    Action: [T1, T2, T3, T4] (4 motor thrusts)
    """
    
    @property
    def name(self) -> str:
        """Environment name identifier."""
        return "drone_full_3d"
    
    def create_env(self, config: Dict[str, Any]) -> Any:
        """
        Create drone_full_3d environment instance.
        
        Args:
            config: Environment configuration dictionary with keys:
                - dt: Time step (default: 0.1)
                - horizon: Planning horizon (default: 80)
                - p_max: Position bounds (default: 2.0)
                - v_max: Velocity bounds (default: 2.0)
                - control_limit: Control limit (default: 1.0)
                - target: Target position [x, y, z] (default: [0.0, 0.0, 1.0])
                - mass, Ixx, Iyy, Izz, arm_length, kf, km, gravity: DroneModel parameters
                - Any other DroneFull3DEnv parameters
                
        Returns:
            DroneFull3DEnv instance
        """
        from genedynamics.envs.factories import make_env
        return make_env(self.name, **config)
    
    def create_energy(self) -> Any:
        """
        Create energy functional for drone_full_3d environment.
        
        Returns:
            LegacyEnergyFunctional instance
        """
        from genedynamics.envs.factories import make_energy
        return make_energy(self.name)
    
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

    def get_position_dim(self) -> int:
        """Position dimension (3 for 3D)."""
        return 3


class DroneFull3DPhysicsPlugin(EnvironmentPlugin):
    """
    Plugin for drone_full_3d_physics environment (unified interface).
    
    This plugin supports multiple physics backends (drone_model, mujoco, isaac)
    and computational backends (numpy, jax) through a unified configuration.
    State: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz] (12D)
    Action: [T1, T2, T3, T4] (4 motor thrusts)
    """
    
    @property
    def name(self) -> str:
        """Environment name identifier."""
        return "drone_full_3d_physics"
    
    def create_env(self, config: Dict[str, Any]) -> Any:
        """
        Create drone_full_3d_physics environment instance.
        
        Args:
            config: Environment configuration dictionary with keys:
                - physics_backend: 'drone_model', 'mujoco', 'mjx', 'isaac', or None
                - renderer: 'matplotlib', 'mujoco', 'isaac', or None
                - use_jax_dynamics: Use JAX dynamics for planning (default: True)
                - model_path_mujoco: Path to MuJoCo XML file (optional)
                - model_path_isaac: Path to Isaac Sim USD file (optional)
                - obstacles: ObstacleManager instance (optional)
                - All other DroneFull3DPhysicsEnv parameters
                
        Returns:
            Environment instance (DroneFull3DPhysicsEnv, DroneFull3DMujocoEnv, or DroneFull3DIsaacEnv)
        """
        from genedynamics.envs.factories import make_env
        
        # Determine which environment to create based on physics_backend
        physics_backend = config.get('physics_backend', 'drone_model')
        
        if physics_backend == 'mujoco':
            env_name = 'drone_full_3d_mujoco'
        elif physics_backend == 'mjx':
            env_name = 'drone_full_3d_mjx'
        elif physics_backend == 'isaac':
            env_name = 'drone_full_3d_isaac'
        else:
            env_name = 'drone_full_3d_physics'
        
        return make_env(env_name, **config)
    
    def create_energy(self) -> Any:
        """
        Create energy functional for drone_full_3d_physics environment.
        
        Returns:
            LegacyEnergyFunctional instance
        """
        from genedynamics.envs.factories import make_energy
        return make_energy('drone_full_3d_physics')
    
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

