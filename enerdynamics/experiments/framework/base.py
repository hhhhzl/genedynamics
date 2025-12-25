"""
Base plugin interfaces for the experimental framework.

This module defines abstract base classes for all plugin types:
- MethodPlugin: Solver methods (EDOC, MPPI, CEM, etc.)
- EnvironmentPlugin: Environments (single/double integrator, drone, etc.)
- MetricsPlugin: Metrics computation (SSR, obstacle density, etc.)
- VisualizationPlugin: Visualization generation
- ObstacleGeneratorPlugin: Obstacle generation strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
from enerdynamics.core.types import Trajectory


class MethodPlugin(ABC):
    """
    Base class for solver method plugins.
    
    Each solver method (EDOC, MPPI, CEM, etc.) should implement this interface
    to be compatible with the experiment framework.
    """
    
    @abstractmethod
    def create_planner(self, env: Any, energy: Any, config: Dict[str, Any]) -> Any:
        """
        Create and configure planner instance.
        
        Args:
            env: Environment instance
            energy: Energy functional instance
            config: Method-specific configuration dictionary
            
        Returns:
            Planner instance ready for planning
        """
        pass
    
    @abstractmethod
    def plan(self, planner: Any, initial_state: np.ndarray, rng: Any) -> Dict[str, Any]:
        """
        Execute planning with the given planner.
        
        Args:
            planner: Planner instance created by create_planner
            initial_state: Initial state for planning
            rng: Random number generator
            
        Returns:
            Dictionary containing planning results:
            - states: List of states
            - actions: List of actions
            - energies: Energy values over trajectory (optional)
            - rewards: Reward values over trajectory (optional)
            - diffusion_actions_traj: Diffusion history (optional)
            - diffusion_sampled_actions: Diffusion samples (optional)
            - initial_state: Initial state used
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Method name identifier.
        
        Returns:
            Method name (e.g., "edoc", "mppi", "cem")
        """
        pass


class EnvironmentPlugin(ABC):
    """
    Base class for environment plugins.
    
    Each environment type should implement this interface to provide
    environment-specific functionality for experiments.
    """
    
    @abstractmethod
    def create_env(self, config: Dict[str, Any]) -> Any:
        """
        Create environment instance.
        
        Args:
            config: Environment configuration dictionary
            
        Returns:
            Environment instance
        """
        pass
    
    @abstractmethod
    def create_energy(self) -> Any:
        """
        Create energy functional for this environment.
        
        Returns:
            Energy functional instance
        """
        pass
    
    @abstractmethod
    def get_state_dim(self) -> int:
        """
        Get state dimension.
        
        Returns:
            State dimension
        """
        pass
    
    @abstractmethod
    def extract_position(self, state: np.ndarray) -> np.ndarray:
        """
        Extract position coordinates from state for visualization.
        
        For 2D environments, this should return (x, y).
        For 3D environments, this should return (x, y, z).
        
        Args:
            state: Full state vector
            
        Returns:
            Position vector (2D or 3D)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Environment name identifier.
        
        Returns:
            Environment name (e.g., "single_integrator_box_2d")
        """
        pass


class MetricsPlugin(ABC):
    """
    Base class for metrics computation plugins.
    
    Each metric type should implement this interface to compute
    metrics from experiment results.
    """
    
    @abstractmethod
    def compute(self, trajectory: Trajectory, env: Any, obstacles: Any, 
                constraints: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Compute metrics from trajectory and environment.
        
        Args:
            trajectory: Computed trajectory
            env: Environment instance
            obstacles: Obstacle manager instance
            constraints: Constraint manager instance (optional)
            **kwargs: Additional parameters (robot_radius, success_margin, etc.)
            
        Returns:
            Dictionary containing metric values (all values should be JSON-serializable)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Metric name identifier.
        
        Returns:
            Metric name (e.g., "ssr", "obstacle_density", "nonconvexity")
        """
        pass


class VisualizationPlugin(ABC):
    """
    Base class for visualization plugins.
    
    Each visualization type should implement this interface to generate
    plots and figures from experiment results.
    """
    
    @abstractmethod
    def visualize(self, fig: Any, ax: Any, data: Dict[str, Any], 
                  config: Dict[str, Any]) -> None:
        """
        Create visualization on the given axes.
        
        Args:
            fig: Matplotlib figure
            ax: Matplotlib axes (or array of axes)
            data: Dictionary containing data to visualize:
                - trajectory: Trajectory object
                - env: Environment instance
                - obstacles: Obstacle manager
                - result: Planning result dictionary
                - Any other relevant data
            config: Visualization configuration dictionary
        """
        pass
    
    @abstractmethod
    def save(self, output_path: Path, fig: Any, **kwargs: Any) -> None:
        """
        Save visualization to file.
        
        Args:
            output_path: Path to save the figure
            fig: Matplotlib figure
            **kwargs: Additional save options (dpi, bbox_inches, etc.)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Visualization name identifier.
        
        Returns:
            Visualization name (e.g., "trajectory", "diffusion", "energy_reward")
        """
        pass


class ObstacleGeneratorPlugin(ABC):
    """
    Base class for obstacle generation plugins.
    
    Each obstacle generation strategy should implement this interface
    to generate obstacle configurations for experiments.
    """
    
    @abstractmethod
    def generate(self, level: int, seed: int, start_pos: np.ndarray,
                 target_pos: np.ndarray, config: Dict[str, Any]) -> Any:
        """
        Generate obstacles for given level and configuration.
        
        Args:
            level: Obstacle difficulty level (0-10, typically)
            seed: Random seed for reproducibility
            start_pos: Start position
            target_pos: Target position
            config: Obstacle generation configuration:
                - robot_radius: Robot radius for collision checking
                - map_bounds: Dictionary with x_min, x_max, y_min, y_max
                - Any other generator-specific parameters
                
        Returns:
            ObstacleManager instance with generated obstacles
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Generator name identifier.
        
        Returns:
            Generator name (e.g., "box2d", "drone3d")
        """
        pass

