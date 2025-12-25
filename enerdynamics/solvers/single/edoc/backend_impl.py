"""
Backend implementations for EDOC solver.

This module defines the abstract base class and protocol for EDOC backend implementations.
Each backend (JAX, NumPy) provides its own implementation of the reverse diffusion algorithm.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Optional, Any, Dict, Tuple
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None


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
        self.horizon = planner.horizon
        self.action_diffuse_steps = planner.action_diffuse_steps
        self.action_beta0 = planner.action_beta0
        self.action_betaT = planner.action_betaT
        self.action_temp = planner.action_temp
        self.action_extra_sigma = planner.action_extra_sigma
        self.action_score_mode = planner.action_score_mode
        self.action_nsample = planner.action_nsample
        self.use_antithetic = planner.use_antithetic
        self.constraint_manager = planner.constraint_manager
        self.lambda_energy = planner.lambda_energy
        self.use_constraint_in_scoring = planner.use_constraint_in_scoring
        self.terminal_energy_weight = planner.terminal_energy_weight
        self.show_tqdm = planner.show_tqdm
        self._reverse_diffuse_chunk_len = planner._reverse_diffuse_chunk_len
    
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
    
    def initialize(self) -> None:
        """
        Initialize backend-specific resources (e.g., JIT-compiled functions).
        
        This is called after __init__ to set up any backend-specific state.
        Subclasses can override this to perform initialization.
        """
        pass

