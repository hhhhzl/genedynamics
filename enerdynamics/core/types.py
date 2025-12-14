"""
Core type definitions and protocols for the energy-driven control framework.

This module defines the fundamental abstractions that all components build upon:
- State: represents system state (can be PyTree, torch.Tensor dict, numpy array, etc.)
- Action: represents control actions
- Trajectory: represents a sequence of states and actions

These types are intentionally flexible to support multiple backends (JAX, PyTorch, Rust, etc.)
"""

from typing import Any, Protocol, runtime_checkable
from dataclasses import dataclass
from typing import List, Optional


# State and Action are type aliases for flexibility
# In practice, they can be any array-like structure (numpy, JAX, PyTorch, etc.)
# The Protocol definitions below are for documentation/type checking purposes

State = Any  # Can be numpy array, JAX array, PyTorch tensor, dict, etc.
Action = Any  # Can be numpy array, JAX array, PyTorch tensor, etc.


# Optional: If you want to enforce a specific structure, you can use these:
@runtime_checkable
class StateProtocol(Protocol):
    """
    Protocol for structured state representation (optional).
    
    If you want to use structured states with metadata, implement this protocol.
    Otherwise, you can use plain arrays (numpy, JAX, PyTorch) as State.
    """
    data: Any


@runtime_checkable
class ActionProtocol(Protocol):
    """
    Protocol for structured action representation (optional).
    
    If you want to use structured actions with metadata, implement this protocol.
    Otherwise, you can use plain arrays (numpy, JAX, PyTorch) as Action.
    """
    data: Any


@dataclass
class Trajectory:
    """
    Represents a sequence of states and actions over time.
    
    This is the fundamental data structure for trajectory optimization,
    energy evaluation, and constraint checking.
    
    Attributes:
        states: List of states [x_0, x_1, ..., x_T]
        actions: List of actions [u_0, u_1, ..., u_{T-1}]
        info: Optional metadata dictionary
    """
    states: List[State]
    actions: List[Action]
    info: Optional[dict] = None
    
    def __len__(self) -> int:
        """Return the number of time steps (length of states list)."""
        return len(self.states)
    
    def __post_init__(self):
        """Validate that trajectory has consistent length."""
        if len(self.states) != len(self.actions) + 1:
            raise ValueError(
                f"Trajectory length mismatch: {len(self.states)} states but "
                f"{len(self.actions)} actions. Expected states = actions + 1."
            )


# Type aliases for convenience
StateType = Any  # Can be State protocol or concrete implementation
ActionType = Any  # Can be Action protocol or concrete implementation
TrajectoryType = Trajectory

