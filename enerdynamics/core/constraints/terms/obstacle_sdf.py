"""
SDF-based obstacle constraint terms.

This module provides obstacle constraints based on signed distance functions (SDF).
Supports margin tightening for schedule-based constraint relaxation.
"""

from typing import Callable, Optional
import numpy as np

from .base import ConstraintTerm
from enerdynamics.core.types import Trajectory, State
from enerdynamics.envs.obstacles.base import ObstacleManager


class ObstacleSDFTerm(ConstraintTerm):
    """
    Obstacle constraint term based on signed distance function (SDF).
    
    This term defines obstacle avoidance constraints:
    - Energy: Barrier energy exp(-β * (sdf - margin))
    - Feasible: sdf >= margin
    - Violation: max(0, margin - sdf)
    
    The margin parameter can be scheduled to gradually tighten constraints
    (soft → hard transition).
    """
    
    def __init__(
        self,
        obstacles: ObstacleManager,
        margin: float = 0.0,
        beta: float = 10.0,
        position_extractor: Optional[Callable[[State], np.ndarray]] = None,
    ):
        """
        Initialize obstacle SDF constraint term.
        
        Args:
            obstacles: ObstacleManager containing obstacles
            margin: Safety margin (constraint: sdf >= margin)
            beta: Barrier sharpness (for energy computation)
            position_extractor: Function to extract position from state
        """
        self.obstacles = obstacles
        self.margin = margin
        self.beta = beta
        self.position_extractor = position_extractor or self._default_extract_position
    
    def energy(
        self,
        trajectory: Trajectory,
        margin: Optional[float] = None,
        **kwargs
    ) -> float:
        """
        Compute barrier energy: Σ_h exp(-β * (sdf(x^h) - margin)).
        
        Args:
            trajectory: Trajectory to evaluate
            margin: Override default margin (for scheduling)
            **kwargs: Additional parameters
            
        Returns:
            Total barrier energy
        """
        margin = margin if margin is not None else self.margin
        total = 0.0
        
        for state in trajectory.states:
            pos = self.position_extractor(state)
            sdf = self.obstacles.sdf(pos)
            sdf = float(np.asarray(sdf).item() if hasattr(sdf, 'item') else sdf)
            
            # Barrier energy: exp(-β * (sdf - margin))
            # High when sdf < margin (close to obstacles)
            clearance = sdf - margin
            total += np.exp(-self.beta * clearance)
        
        return float(total)
    
    def feasible(
        self,
        trajectory: Trajectory,
        margin: Optional[float] = None,
        **kwargs
    ) -> bool:
        """
        Check if trajectory satisfies obstacle constraints: sdf >= margin.
        
        Args:
            trajectory: Trajectory to check
            margin: Override default margin (for scheduling)
            **kwargs: Additional parameters
            
        Returns:
            True if all states satisfy sdf >= margin
        """
        margin = margin if margin is not None else self.margin
        
        for state in trajectory.states:
            pos = self.position_extractor(state)
            sdf = self.obstacles.sdf(pos)
            sdf = float(np.asarray(sdf).item() if hasattr(sdf, 'item') else sdf)
            
            if sdf < margin:
                return False
        
        return True
    
    def violation(
        self,
        trajectory: Trajectory,
        margin: Optional[float] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Compute constraint violations: max(0, margin - sdf) for each state.
        
        Args:
            trajectory: Trajectory to evaluate
            margin: Override default margin (for scheduling)
            **kwargs: Additional parameters
            
        Returns:
            Array of violations per state, shape (H+1,)
        """
        margin = margin if margin is not None else self.margin
        violations = []
        
        for state in trajectory.states:
            pos = self.position_extractor(state)
            sdf = self.obstacles.sdf(pos)
            sdf = float(np.asarray(sdf).item() if hasattr(sdf, 'item') else sdf)
            
            # Violation: max(0, margin - sdf)
            violation = max(0.0, margin - sdf)
            violations.append(violation)
        
        return np.array(violations, dtype=np.float32)
    
    def energy_batch(
        self,
        trajectories: list[Trajectory],
        margin: Optional[float] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Batch compute energy for multiple trajectories.
        
        Optimized version that extracts all positions and computes SDF in batch.
        
        Args:
            trajectories: List of trajectories
            margin: Override default margin
            **kwargs: Additional parameters
            
        Returns:
            Array of energy values, shape (len(trajectories),)
        """
        margin = margin if margin is not None else self.margin
        
        # Extract all positions
        all_positions = []
        traj_lengths = []
        for traj in trajectories:
            positions = [self.position_extractor(state) for state in traj.states]
            all_positions.extend(positions)
            traj_lengths.append(len(traj.states))
        
        if not all_positions:
            return np.zeros(len(trajectories), dtype=np.float32)
        
        # Batch compute SDF
        positions_array = np.stack(all_positions, axis=0).astype(np.float32)
        sdfs = self.obstacles.sdf(positions_array)
        
        if not isinstance(sdfs, np.ndarray):
            sdfs = np.asarray(sdfs)
        if sdfs.ndim == 0:
            sdfs = sdfs.reshape(1)
        
        # Compute barrier energy for each position
        clearances = sdfs - margin
        energies_per_position = np.exp(-self.beta * clearances)
        
        # Sum over each trajectory
        energies = []
        idx = 0
        for length in traj_lengths:
            traj_energy = np.sum(energies_per_position[idx:idx+length])
            energies.append(float(traj_energy))
            idx += length
        
        return np.array(energies, dtype=np.float32)
    
    @staticmethod
    def _default_extract_position(state: State) -> np.ndarray:
        """Default position extractor."""
        state_np = np.asarray(state, dtype=np.float32)
        if len(state_np) == 4:  # 2D double integrator
            return state_np[:2]
        elif len(state_np) == 2:  # 1D double integrator
            return state_np[:1]
        return state_np[:min(2, len(state_np))]

