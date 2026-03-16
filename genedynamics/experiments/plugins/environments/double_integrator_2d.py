"""
Double integrator 2D environment plugin.
"""

from typing import Dict, Any
import numpy as np

from genedynamics.envs.factories import make_env, make_energy
from ...framework.base import EnvironmentPlugin


class DoubleIntegrator2DPlugin(EnvironmentPlugin):
    """
    Plugin for double integrator 2D environment.
    
    This environment represents a 2D point robot with position and velocity dynamics.
    """
    
    @property
    def name(self) -> str:
        """Environment name identifier."""
        return "double_integrator_box_2d"
    
    def create_env(self, config: Dict[str, Any]) -> Any:
        """
        Create double integrator 2D environment instance.
        
        Args:
            config: Environment configuration dictionary with keys:
                - dt: Time step (default: 0.1)
                - horizon: Planning horizon (default: 80)
                - p_max: Position bounds (default: 2.0)
                - v_max: Velocity bounds (default: 2.0)
                - control_limit: Control limit (default: 1.0)
                - Any other DoubleIntegratorBox2DEnv parameters
                
        Returns:
            DoubleIntegratorBox2DEnv instance
        """
        return make_env(self.name, **config)
    
    def create_energy(self) -> Any:
        """
        Create energy functional for double integrator 2D environment.
        
        Returns:
            LegacyEnergyFunctional instance
        """
        return make_energy(self.name)
    
    def get_state_dim(self) -> int:
        """
        Get state dimension.
        
        Returns:
            State dimension (4 for position and velocity: x, y, vx, vy)
        """
        return 4
    
    def extract_position(self, state: np.ndarray) -> np.ndarray:
        """
        Extract position coordinates from state.
        
        Args:
            state: Full state vector (x, y, vx, vy)
            
        Returns:
            Position vector (x, y)
        """
        return np.asarray(state, dtype=np.float32)[:2]

