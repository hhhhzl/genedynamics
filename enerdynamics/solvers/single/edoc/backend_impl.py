"""
Backend implementations for EDOC solver.

This module defines the abstract base class and protocol for EDOC backend implementations.
Each backend (JAX, NumPy) provides its own implementation of the reverse diffusion algorithm.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Optional, Any, Dict, Tuple, List
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

from enerdynamics.core.types import Trajectory


class EDOCBackend(Protocol):
    """
    Protocol for EDOC backend implementations.
    Defines the interface that all backend-specific EDOC classes must adhere to.
    """
    
    def reverse_diffuse(
        self,
        rng_key: Any,
        state_init: np.ndarray,
        hard_clearance_by_idx: Optional[np.ndarray],
        hard_enabled_by_idx: Optional[np.ndarray],
        soft_alpha_by_idx: Optional[np.ndarray],
        soft_beta_by_idx: Optional[np.ndarray],
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run reverse diffusion algorithm.
        
        Args:
            rng_key: Random number generator key
            state_init: Initial state, shape (state_dim,)
            hard_clearance_by_idx: Hard clearance per diffusion step, shape (Ndiffuse,)
            hard_enabled_by_idx: Hard enabled flag per diffusion step, shape (Ndiffuse,)
            soft_alpha_by_idx: Soft alpha parameter per diffusion step, shape (Ndiffuse,)
            soft_beta_by_idx: Soft beta parameter per diffusion step, shape (Ndiffuse,)
        
        Returns:
            Tuple of (rng_key_out, actions, reward_history, trajectory_history, samples_history)
            - rng_key_out: Updated RNG key
            - actions: Final actions, shape (horizon, act_dim)
            - reward_history: Reward history, shape (Ndiffuse-1,)
            - trajectory_history: Trajectory history, shape (Ndiffuse-1, horizon, act_dim)
            - samples_history: Samples history, shape (Ndiffuse-1, num_particles, horizon, act_dim)
        """
        ...


class EDOCBackendBase(ABC):
    """
    Base class for EDOC backend implementations.
    Provides common initialization and utility methods.
    """
    
    def __init__(self, planner: Any, **config: Any):
        """
        Initialize EDOC backend implementation.
        
        Args:
            planner: EDOCPlanner instance (for accessing configuration and methods)
            **config: Additional backend-specific configuration
        """
        self.planner = planner
        # Extract common configuration from planner
        self.env = planner.env
        self.energy = planner.energy
        self.horizon = planner.horizon
        self.action_space = planner.action_space
        self.action_diffuse_steps = planner.action_diffuse_steps
        self.action_beta0 = planner.action_beta0
        self.action_betaT = planner.action_betaT
        self.action_temp = planner.action_temp
        self.action_extra_sigma = planner.action_extra_sigma
        self.action_score_mode = planner.action_score_mode
        self.action_nsample = planner.action_nsample
        self.use_antithetic = planner.use_antithetic
        self.constraint_manager = planner.constraint_manager
        self.constraint_pipeline = getattr(planner, 'constraint_pipeline', None)
        self.lambda_energy = planner.lambda_energy
        self.use_constraint_in_scoring = planner.use_constraint_in_scoring
        self.terminal_energy_weight = planner.terminal_energy_weight
        self.guide_weight = getattr(planner, 'guide_weight', 0.0)
        self.dyn_loss_coeff = planner.dyn_loss_coeff
        self.dyn_loss_mode = planner.dyn_loss_mode
        self.show_tqdm = planner.show_tqdm
        self._reverse_diffuse_chunk_len = planner._reverse_diffuse_chunk_len
        self.scheduler = getattr(planner, 'scheduler', None)
    
    @abstractmethod
    def reverse_diffuse(
        self,
        rng_key: Any,
        state_init: np.ndarray,
        hard_clearance_by_idx: Optional[np.ndarray],
        hard_enabled_by_idx: Optional[np.ndarray],
        soft_alpha_by_idx: Optional[np.ndarray],
        soft_beta_by_idx: Optional[np.ndarray],
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run reverse diffusion algorithm.
        
        Args:
            rng_key: Random number generator key
            state_init: Initial state, shape (state_dim,)
            hard_clearance_by_idx: Hard clearance per diffusion step, shape (Ndiffuse,)
            hard_enabled_by_idx: Hard enabled flag per diffusion step, shape (Ndiffuse,)
            soft_alpha_by_idx: Soft alpha parameter per diffusion step, shape (Ndiffuse,)
            soft_beta_by_idx: Soft beta parameter per diffusion step, shape (Ndiffuse,)
        
        Returns:
            Tuple of (rng_key_out, actions, reward_history, trajectory_history, samples_history)
            - rng_key_out: Updated RNG key
            - actions: Final actions, shape (horizon, act_dim)
            - reward_history: Reward history, shape (Ndiffuse-1,)
            - trajectory_history: Trajectory history, shape (Ndiffuse-1, horizon, act_dim)
            - samples_history: Samples history, shape (Ndiffuse-1, num_particles, horizon, act_dim)
        """
        pass
    
    @abstractmethod
    def rollout_states_and_energy(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        hard_clearance: float,
        hard_enabled: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rollout states and compute energy sequence.
        
        Args:
            state: Initial state, shape (state_dim,)
            actions: Action sequence, shape (horizon, act_dim)
            hard_clearance: Hard constraint clearance
            hard_enabled: Whether hard constraints are enabled
        
        Returns:
            Tuple of (states_full, energy_seq)
            - states_full: Full state trajectory, shape (horizon+1, state_dim)
            - energy_seq: Energy sequence, shape (horizon,)
        """
        pass
    
    @abstractmethod
    def rollout_env_states(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        hard_clearance: float,
        hard_enabled: bool,
    ) -> np.ndarray:
        """
        Rollout environment states.
        
        Args:
            state: Initial state, shape (state_dim,)
            actions: Action sequence, shape (horizon, act_dim)
            hard_clearance: Hard constraint clearance
            hard_enabled: Whether hard constraints are enabled
        
        Returns:
            Full state trajectory, shape (horizon+1, state_dim)
        """
        pass
    
    @abstractmethod
    def compute_total_energy(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        hard_clearance: float,
        hard_enabled: bool,
    ) -> float:
        """
        Compute total energy including terminal and dynamics loss.
        
        Args:
            state: Initial state, shape (state_dim,)
            actions: Action sequence, shape (horizon, act_dim)
            hard_clearance: Hard constraint clearance
            hard_enabled: Whether hard constraints are enabled
        
        Returns:
            Total energy value (scalar)
        """
        pass
    
    @abstractmethod
    def compute_mean_reward(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        hard_clearance: float,
        hard_enabled: bool,
    ) -> float:
        """
        Compute mean reward over trajectory.
        
        Args:
            state: Initial state, shape (state_dim,)
            actions: Action sequence, shape (horizon, act_dim)
            hard_clearance: Hard constraint clearance
            hard_enabled: Whether hard constraints are enabled
        
        Returns:
            Mean reward value (scalar)
        """
        pass
    
    @abstractmethod
    def score_particles(
        self,
        state_init: np.ndarray,
        batch_actions: np.ndarray,
        step: Optional[int],
        total_steps: Optional[int],
    ) -> np.ndarray:
        """
        Score action particles in batch.
        
        Args:
            state_init: Initial state, shape (state_dim,)
            batch_actions: Batch of action sequences, shape (num_particles, horizon, act_dim)
            step: Current diffusion step
            total_steps: Total diffusion steps
        
        Returns:
            Scores array, shape (num_particles,)
        """
        pass
    
    @abstractmethod
    def actions_to_trajectory(
        self,
        x0: np.ndarray,
        actions: np.ndarray,
    ) -> Trajectory:
        """
        Convert actions array to Trajectory for constraint evaluation.
        
        Args:
            x0: Initial state, shape (state_dim,)
            actions: Action sequence, shape (horizon, act_dim) or single action (act_dim)
        
        Returns:
            Trajectory object
        """
        pass
    
    @abstractmethod
    def extract_actions_from_trajectory(
        self,
        trajectory: Trajectory,
    ) -> np.ndarray:
        """
        Extract actions array from trajectory.
        
        Args:
            trajectory: Trajectory object
        
        Returns:
            Actions array, shape (horizon, act_dim)
        """
        pass
    
    @abstractmethod
    def track_actions_to_projected_states(
        self,
        x0: np.ndarray,
        actions_init: np.ndarray,
        target_states: List[np.ndarray],
        gn_iters: int = 2,
        reg: float = 1e-3,
    ) -> np.ndarray:
        """
        Map projected state trajectory back to actions using Gauss-Newton tracking.
        
        Args:
            x0: Initial state, shape (state_dim,)
            actions_init: Initial actions, shape (horizon, act_dim)
            target_states: List of target states, length (horizon+1)
            gn_iters: Number of Gauss-Newton iterations
            reg: Regularization for tracking
        
        Returns:
            Updated actions, shape (horizon, act_dim)
        """
        pass
    
    @abstractmethod
    def add_extra_noise(
        self,
        actions_array: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """
        Add extra noise to actions.
        
        Args:
            actions_array: Actions array, shape (horizon, act_dim)
            sigma: Noise standard deviation
        
        Returns:
            Noisy actions array, shape (horizon, act_dim)
        """
        pass
    
    def initialize(self) -> None:
        """
        Initialize backend-specific resources (e.g., JIT-compiled functions).
        
        This is called after __init__ to set up any backend-specific state.
        Subclasses can override this to perform initialization.
        """
        pass

