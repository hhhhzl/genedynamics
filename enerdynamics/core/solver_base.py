"""
Solver base class for the energy-driven control framework.

This module defines the Solver abstract interface, which represents any
algorithm that can solve the energy minimization problem:
    minimize E(trajectory) subject to dynamics constraints

All control/RL/world model methods can be unified under this interface:
- EDOC: diffusion-based solver with A-MCSA + multi-scale + ADM
- MBD / EB-MBD / MDOC: barrier-based solvers
- MPPI / iLQR / PETS: sampling/optimization-based solvers
- SAC / RL: policy-based solvers (policy rollout = solve)
- Flow Matching: learned dynamics + energy minimization

The key insight is that all these methods take (dynamics, energy) and
produce a trajectory, so they can share the same interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from enerdynamics.core.types import State, Trajectory
from enerdynamics.core.dynamics import DynamicsModel
from enerdynamics.core.energy import EnergyFunctional
from enerdynamics.core.backend import Backend


class Solver(ABC):
    """
    Abstract base class for trajectory optimization solvers.
    
    A solver takes a dynamics model and energy functional, and produces
    an approximately optimal trajectory from an initial state.
    
    All solvers follow the same interface:
    1. Initialize with (dynamics, energy, backend)
    2. solve(x0, horizon, **kwargs) -> trajectory
    
    This allows easy swapping of different algorithms without changing
    the rest of the codebase.
    """
    
    def __init__(
        self,
        dynamics: DynamicsModel,
        energy: EnergyFunctional,
        backend: Backend,
        **kwargs
    ):
        """
        Initialize solver with dynamics, energy, and backend.
        
        Args:
            dynamics: Dynamics model (physics or learned)
            energy: Energy functional (task + constraints)
            backend: Computational backend (JAX, PyTorch, etc.)
            **kwargs: Solver-specific configuration
        """
        self.dynamics = dynamics
        self.energy = energy
        self.backend = backend
        self.config = kwargs
    
    @abstractmethod
    def solve(
        self,
        x0: State,
        horizon: int,
        **kwargs
    ) -> Trajectory:
        """
        Solve for an approximately optimal trajectory.
        
        This is the main interface that all solvers must implement.
        The returned trajectory should minimize the energy functional
        while respecting the dynamics constraints.
        
        Args:
            x0: Initial state
            horizon: Planning horizon (number of time steps)
            **kwargs: Solver-specific options (e.g., number of iterations,
                     temperature, etc.)
            
        Returns:
            Trajectory: Approximately optimal trajectory
        """
        pass
    
    def rollout(self, x0: State, actions: list) -> Trajectory:
        """
        Helper method: roll out actions using dynamics model.
        
        Args:
            x0: Initial state
            actions: List of actions
            
        Returns:
            Trajectory from rolling out actions
        """
        return self.dynamics.rollout(x0, actions)
    
    def evaluate(self, traj: Trajectory) -> Dict[str, float]:
        """
        Evaluate a trajectory using the energy functional.
        
        Returns a breakdown of task cost, constraint energy, and total energy.
        
        Args:
            traj: Trajectory to evaluate
            
        Returns:
            Dictionary with energy breakdown
        """
        return {
            "task_cost": self.energy.task_cost(traj),
            "constraint_energy": self.energy.constraint_energy(traj),
            "entropy_term": self.energy.entropy_term(traj),
            "total_energy": self.energy.total_energy(traj),
        }


class SamplingSolver(Solver):
    """
    Base class for sampling-based solvers (e.g., MPPI, CEM, EDOC).
    
    These solvers generate candidate trajectories and select the best ones.
    """
    
    @abstractmethod
    def sample_trajectories(
        self,
        x0: State,
        horizon: int,
        n_samples: int,
        **kwargs
    ) -> list[Trajectory]:
        """
        Sample candidate trajectories.
        
        Args:
            x0: Initial state
            horizon: Planning horizon
            n_samples: Number of trajectories to sample
            **kwargs: Sampling parameters
            
        Returns:
            List of candidate trajectories
        """
        pass
    
    def solve(
        self,
        x0: State,
        horizon: int,
        n_samples: int = 100,
        **kwargs
    ) -> Trajectory:
        """
        Solve by sampling trajectories and selecting the best.
        
        Args:
            x0: Initial state
            horizon: Planning horizon
            n_samples: Number of samples to generate
            **kwargs: Additional solver parameters
            
        Returns:
            Best trajectory from samples
        """
        candidates = self.sample_trajectories(x0, horizon, n_samples, **kwargs)
        
        # Evaluate all candidates
        best_traj = None
        best_energy = float('inf')
        for traj in candidates:
            energy = self.energy.total_energy(traj)
            if energy < best_energy:
                best_energy = energy
                best_traj = traj
        
        return best_traj if best_traj is not None else candidates[0]


class OptimizationSolver(Solver):
    """
    Base class for gradient-based optimization solvers (e.g., iLQR, DDP).
    
    These solvers use iterative optimization to refine trajectories.
    """
    
    @abstractmethod
    def optimize(
        self,
        x0: State,
        horizon: int,
        initial_traj: Optional[Trajectory] = None,
        **kwargs
    ) -> Trajectory:
        """
        Optimize trajectory using gradient-based methods.
        
        Args:
            x0: Initial state
            horizon: Planning horizon
            initial_traj: Optional initial guess
            **kwargs: Optimization parameters (learning rate, iterations, etc.)
            
        Returns:
            Optimized trajectory
        """
        pass
    
    def solve(
        self,
        x0: State,
        horizon: int,
        **kwargs
    ) -> Trajectory:
        """Solve by calling optimize()."""
        return self.optimize(x0, horizon, **kwargs)

