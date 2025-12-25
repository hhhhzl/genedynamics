"""
Single integrator 2D environment plugin.
"""

from typing import Dict, Any
import numpy as np

from enerdynamics.envs.factories import make_env, make_energy
from ...framework.base import EnvironmentPlugin


class SingleIntegrator2DPlugin(EnvironmentPlugin):
    """
    Plugin for single integrator 2D environment.
    
    This environment represents a 2D point robot with position-only dynamics.
    """
    
    @property
    def name(self) -> str:
        """Environment name identifier."""
        return "single_integrator_box_2d"
    
    def create_env(self, config: Dict[str, Any]) -> Any:
        """
        Create single integrator 2D environment instance.
        
        Args:
            config: Environment configuration dictionary with keys:
                - dt: Time step (default: 0.1)
                - horizon: Planning horizon (default: 80)
                - p_max: Position bounds (default: 2.0)
                - control_limit: Control limit (default: 1.0)
                - Any other SingleIntegratorBox2DEnv parameters
                
        Returns:
            SingleIntegratorBox2DEnv instance
        """
        return make_env(self.name, **config)
    
    def create_energy(self) -> Any:
        """
        Create energy functional for single integrator 2D environment.
        
        Returns:
            LegacyEnergyFunctional instance
        """
        return make_energy(self.name)
    
    def get_state_dim(self) -> int:
        """
        Get state dimension.
        
        Returns:
            State dimension (2 for position only)
        """
        return 2
    
    def extract_position(self, state: np.ndarray) -> np.ndarray:
        """
        Extract position coordinates from state.
        
        Args:
            state: Full state vector (x, y)
            
        Returns:
            Position vector (x, y)
        """
        return np.asarray(state, dtype=np.float32)[:2]

