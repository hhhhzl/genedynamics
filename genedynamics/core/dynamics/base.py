"""
Base dynamics model classes.

This module defines the abstract base classes for dynamics models:
- DynamicsModel: Abstract interface
- DeterministicDynamicsModel: Base for deterministic models
- StochasticDynamicsModel: Base for stochastic models
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from genedynamics.core.types import State, Action, Trajectory


class DynamicsModel(ABC):
    """
    Abstract base class for system dynamics models.
    
    A dynamics model defines the transition function x_{t+1} = f(x_t, u_t).
    This can represent:
    - True physics (e.g., double integrator, quadrotor dynamics)
    - Learned models (e.g., neural network world models, flow matching)
    - Hybrid models (e.g., physics-informed neural networks)
    
    All solvers depend only on this interface, allowing easy swapping of
    dynamics models without changing solver code.
    """
    
    @abstractmethod
    def step(self, x: State, u: Action) -> State:
        """
        Single-step dynamics transition: x_{t+1} = f(x_t, u_t).
        
        Args:
            x: Current state
            u: Control action
            
        Returns:
            Next state x_{t+1}
        """
        pass
    
    def rollout(self, x0: State, actions: List[Action]) -> Trajectory:
        """
        Roll out a sequence of actions from initial state.
        
        This is a default implementation that calls step() repeatedly.
        Subclasses can override this for more efficient batch rollouts
        (e.g., using JAX vmap or parallel execution).
        
        Args:
            x0: Initial state
            actions: List of actions [u_0, u_1, ..., u_{T-1}]
            
        Returns:
            Trajectory containing states and actions
        """
        states = [x0]
        for u in actions:
            x_next = self.step(states[-1], u)
            states.append(x_next)
        
        return Trajectory(states=states, actions=actions)
    
    def rollout_batch(self, x0_batch: List[State], actions_batch: List[List[Action]]) -> List[Trajectory]:
        """
        Roll out multiple trajectories in parallel (optional).
        
        Default implementation processes sequentially. Backends with
        vectorization support (JAX, PyTorch) can override for efficiency.
        
        Args:
            x0_batch: List of initial states
            actions_batch: List of action sequences
            
        Returns:
            List of trajectories
        """
        return [self.rollout(x0, actions) for x0, actions in zip(x0_batch, actions_batch)]


class DeterministicDynamicsModel(DynamicsModel):
    """
    Base class for deterministic dynamics models.
    
    This is a convenience class for models that don't have stochasticity.
    Most physics models fall into this category.
    """
    pass


class StochasticDynamicsModel(DynamicsModel):
    """
    Base class for stochastic dynamics models.
    
    For models with process noise: x_{t+1} = f(x_t, u_t) + w_t
    where w_t is random noise.
    """
    
    @abstractmethod
    def step(self, x: State, u: Action, rng_key: Optional[Any] = None) -> State:
        """
        Single-step stochastic dynamics transition.
        
        Args:
            x: Current state
            u: Control action
            rng_key: Random key/seed for noise generation
            
        Returns:
            Next state (with noise)
        """
        pass
