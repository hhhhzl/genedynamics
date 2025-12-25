"""
Base classes for ActionFilter backend implementations.

This module defines the interface that all backend-specific ActionFilter
implementations must follow. Each backend (NumPy, JAX, PyTorch) implements
this interface to provide optimized action filtering functionality.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Optional
import numpy as np

from enerdynamics.core.types import State
from enerdynamics.envs.obstacles.base import ObstacleManager


class ActionFilterBackend(Protocol):
    """
    Protocol for ActionFilter backend implementations.
    
    This defines the interface that all backend implementations must provide.
    Backend-specific classes implement these methods using their native
    operations (NumPy, JAX, PyTorch, etc.).
    """
    
    @abstractmethod
    def filter_actions(
        self,
        x0: State,
        actions: np.ndarray,
        step: Optional[int],
        total_steps: Optional[int],
        **kwargs
    ) -> np.ndarray:
        """
        Filter actions to ensure safety.
        
        This is the core filtering operation that all backends must implement.
        The implementation should use backend-specific optimizations (e.g., JIT
        compilation for JAX, GPU acceleration, etc.).
        
        Args:
            x0: Initial state
            actions: Actions to filter, shape (T, action_dim)
            step: Current step (for scheduling)
            total_steps: Total steps (for scheduling)
            **kwargs: Additional parameters
            
        Returns:
            Filtered actions, shape (T, action_dim)
        """
        ...


class ActionFilterBase(ABC):
    """
    Base class for ActionFilter backend implementations.
    
    This provides common functionality shared by all backend implementations,
    while allowing backend-specific optimizations in the abstract methods.
    
    Subclasses should implement:
    - filter_actions(): Main action filtering method
    """
    
    def __init__(self, obstacles: ObstacleManager, **config):
        """
        Initialize backend implementation.
        
        Args:
            obstacles: ObstacleManager containing obstacles
            **config: Backend-specific configuration
        """
        self.obstacles = obstacles
        self.config = config
    
    @abstractmethod
    def filter_actions(
        self,
        x0: State,
        actions: np.ndarray,
        step: Optional[int],
        total_steps: Optional[int],
        **kwargs
    ) -> np.ndarray:
        """
        Filter actions to ensure safety.
        
        Args:
            x0: Initial state
            actions: Actions to filter, shape (T, action_dim)
            step: Current step (optional)
            total_steps: Total steps (optional)
            **kwargs: Additional parameters
            
        Returns:
            Filtered actions, shape (T, action_dim)
        """
        pass
    
    def _get_clearance(
        self,
        schedule_manager: Optional[any],
        step: Optional[int],
        total_steps: Optional[int]
    ) -> float:
        """
        Get clearance from schedule manager if available.
        
        Args:
            schedule_manager: ConstraintScheduleManager instance (optional)
            step: Current step
            total_steps: Total steps
            
        Returns:
            Clearance value
        """
        if schedule_manager is None:
            return 0.0
        return float(
            schedule_manager.get_hard_clearance(default=0.0, step=step, total_steps=total_steps)
        )

