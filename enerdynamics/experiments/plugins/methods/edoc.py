"""
EDOC method plugin implementation.
"""

from typing import Dict, Any
import numpy as np

from enerdynamics.solvers.single.edoc import EDOCPlanner
from ...framework.base import MethodPlugin


class EDOCMethodPlugin(MethodPlugin):
    """
    Plugin for EDOC solver method.
    """
    
    @property
    def name(self) -> str:
        """Method name identifier."""
        return "edoc"
    
    def create_planner(self, env: Any, energy: Any, config: Dict[str, Any]) -> EDOCPlanner:
        """
        Create and configure EDOC planner instance.
        
        Args:
            env: Environment instance
            energy: Energy functional instance
            config: Method configuration dictionary with keys:
                - constraint_manager: ConstraintManager instance (optional)
                - horizon: Planning horizon (default: env.horizon)
                - dt: Time step (default: env.dt)
                - action_space: Whether to use action space (default: True)
                - diffusion_mode: Diffusion mode (default: "reverse")
                - action_diffuse_steps: Number of diffusion steps (default: 100)
                - action_nsample: Number of samples per step (default: 256)
                - use_antithetic: Use antithetic sampling (default: True)
                - action_score_mode: Scoring mode (default: "energy")
                - use_constraint_in_scoring: Include constraints in scoring (default: True)
                - lambda_energy: Energy weight (default: 1.0)
                - terminal_energy_weight: Terminal energy weight (default: 0.0)
                - Any other EDOCPlanner parameters
                
        Returns:
            Configured EDOCPlanner instance
        """
        horizon = config.get('horizon', getattr(env, 'horizon', 80))
        dt = config.get('dt', getattr(env, 'dt', 0.1))
        
        planner = EDOCPlanner(
            env=env,
            energy=energy,
            horizon=horizon,
            dt=dt,
            action_space=config.get('action_space', True),
            diffusion_mode=config.get('diffusion_mode', 'reverse'),
            action_diffuse_steps=config.get('action_diffuse_steps', 100),
            action_nsample=config.get('action_nsample', 256),
            use_antithetic=config.get('use_antithetic', True),
            action_score_mode=config.get('action_score_mode', 'energy'),
            constraint_manager=config.get('constraint_manager'),
            constraint_pipeline=config.get('constraint_pipeline'),
            scheduler=config.get('scheduler'),  # New scheduler system
            use_constraint_in_scoring=config.get('use_constraint_in_scoring', True),
            lambda_energy=config.get('lambda_energy', 1.0),
            terminal_energy_weight=config.get('terminal_energy_weight', 0.0),
        )
        
        return planner
    
    def plan(self, planner: EDOCPlanner, initial_state: np.ndarray, rng: Any) -> Dict[str, Any]:
        """
        Execute EDOC planning with custom initial state.
        
        Args:
            planner: EDOCPlanner instance
            initial_state: Initial state for planning
            rng: Random number generator
            
        Returns:
            Dictionary containing planning results
        """
        # Temporarily override env.reset to use custom initial state
        original_reset = planner.env.reset
        
        def custom_reset(rng=None):
            return initial_state.copy(), {}
        
        planner.env.reset = custom_reset
        
        try:
            result = planner.plan(rng)
            if result is None:
                raise ValueError("EDOCPlanner.plan() returned None. Planning may have failed.")
            return result
        except Exception as e:
            # Re-raise with context
            raise RuntimeError(f"EDOC planning failed: {e}") from e
        finally:
            # Restore original reset
            planner.env.reset = original_reset

