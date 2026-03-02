"""
Utility functions for constraint system.

This module provides helper functions for:
- topK/topL selection: Active constraint and time step selection
- Batching helpers: Trajectory batching and unbatching
- Array utilities: Backend-agnostic array operations
"""

from typing import List, Union, Tuple, Optional, Any
import numpy as np

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jnp = None
    JAX_AVAILABLE = False

from genedynamics.core.types import Trajectory


def topK_selection(
    values: Union[np.ndarray, List[float]],
    K: int,
    largest: bool = True
) -> np.ndarray:
    """
    Select top-K values (for active constraint selection).
    
    This is used to select the K most important constraints to enforce,
    reducing computational cost while maintaining constraint satisfaction.
    
    Args:
        values: Array of values (e.g., violation magnitudes, SDF values)
        K: Number of top values to select
        largest: If True, select largest; if False, select smallest
        
    Returns:
        Boolean mask indicating selected values (True = selected)
        
    Example:
        >>> violations = np.array([0.1, 0.5, 0.2, 0.8, 0.3])
        >>> mask = topK_selection(violations, K=2, largest=True)
        >>> mask
        array([False, True, False, True, False])
    """
    values = np.asarray(values, dtype=np.float32)
    K = min(K, len(values))
    
    if K == 0:
        return np.zeros(len(values), dtype=bool)
    
    if largest:
        # Select K largest values
        indices = np.argpartition(values, -K)[-K:]
    else:
        # Select K smallest values
        indices = np.argpartition(values, K)[:K]
    
    mask = np.zeros(len(values), dtype=bool)
    mask[indices] = True
    return mask


def topL_selection(
    values: Union[np.ndarray, List[float]],
    L: int,
    largest: bool = True
) -> np.ndarray:
    """
    Select top-L values (for time step selection).
    
    This is used to select the L most critical time steps to repair,
    enabling selective constraint enforcement.
    
    Args:
        values: Array of values (e.g., violation per time step)
        L: Number of top values to select
        largest: If True, select largest; if False, select smallest
        
    Returns:
        Boolean mask indicating selected values (True = selected)
        
    Example:
        >>> violations_per_step = np.array([0.1, 0.5, 0.2, 0.8, 0.3])
        >>> mask = topL_selection(violations_per_step, L=2, largest=True)
        >>> mask
        array([False, True, False, True, False])
    """
    return topK_selection(values, L, largest)


def select_active_constraints(
    violations: Union[np.ndarray, List[float]],
    topK: Optional[int] = None,
    threshold: Optional[float] = None
) -> np.ndarray:
    """
    Select active constraints based on topK or threshold.
    
    Args:
        violations: Array of constraint violations
        topK: Select top-K most violated constraints (if provided)
        threshold: Select constraints with violation > threshold (if provided)
        
    Returns:
        Boolean mask indicating active constraints
        
    Example:
        >>> violations = np.array([0.1, 0.5, 0.2, 0.8, 0.3])
        >>> mask = select_active_constraints(violations, topK=2)
        >>> mask
        array([False, True, False, True, False])
    """
    violations = np.asarray(violations, dtype=np.float32)
    
    if topK is not None:
        return topK_selection(violations, topK, largest=True)
    elif threshold is not None:
        return violations > threshold
    else:
        # Select all non-zero violations
        return violations > 1e-8


def select_active_time_steps(
    violations_per_step: Union[np.ndarray, List[float]],
    topL: Optional[int] = None,
    threshold: Optional[float] = None
) -> np.ndarray:
    """
    Select active time steps based on topL or threshold.
    
    Args:
        violations_per_step: Array of violations per time step
        topL: Select top-L most violated time steps (if provided)
        threshold: Select time steps with violation > threshold (if provided)
        
    Returns:
        Boolean mask indicating active time steps
        
    Example:
        >>> violations = np.array([0.1, 0.5, 0.2, 0.8, 0.3])
        >>> mask = select_active_time_steps(violations, topL=2)
        >>> mask
        array([False, True, False, True, False])
    """
    return select_active_constraints(violations_per_step, topK=topL, threshold=threshold)


def batch_trajectories(
    trajectories: List[Trajectory],
    pad_to_same_length: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Batch trajectories into arrays for efficient processing.
    
    Args:
        trajectories: List of trajectories to batch
        pad_to_same_length: If True, pad trajectories to same length
        
    Returns:
        Tuple of (states_batch, actions_batch, lengths)
        - states_batch: (batch_size, max_H+1, state_dim) array
        - actions_batch: (batch_size, max_H, action_dim) array
        - lengths: List of actual trajectory lengths
    """
    if len(trajectories) == 0:
        return np.zeros((0, 0, 0)), np.zeros((0, 0, 0)), []
    
    # Get dimensions
    state_dim = len(trajectories[0].states[0]) if trajectories[0].states else 0
    action_dim = len(trajectories[0].actions[0]) if trajectories[0].actions else 0
    
    # Get lengths
    lengths = [len(traj.states) for traj in trajectories]
    max_H = max(lengths) if pad_to_same_length else max(lengths)
    
    # Initialize batch arrays
    batch_size = len(trajectories)
    states_batch = np.zeros((batch_size, max_H, state_dim), dtype=np.float32)
    actions_batch = np.zeros((batch_size, max_H - 1, action_dim), dtype=np.float32)
    
    # Fill batch arrays
    for i, traj in enumerate(trajectories):
        H = len(traj.states)
        for t in range(H):
            states_batch[i, t] = np.asarray(traj.states[t], dtype=np.float32)
        for t in range(H - 1):
            actions_batch[i, t] = np.asarray(traj.actions[t], dtype=np.float32)
    
    return states_batch, actions_batch, lengths


def unbatch_trajectories(
    states_batch: np.ndarray,
    actions_batch: np.ndarray,
    lengths: List[int]
) -> List[Trajectory]:
    """
    Unbatch trajectory arrays back into list of trajectories.
    
    Args:
        states_batch: (batch_size, max_H, state_dim) array
        actions_batch: (batch_size, max_H-1, action_dim) array
        lengths: List of actual trajectory lengths
        
    Returns:
        List of Trajectory objects
    """
    trajectories = []
    
    for i, H in enumerate(lengths):
        states = [states_batch[i, t] for t in range(H)]
        actions = [actions_batch[i, t] for t in range(H - 1)]
        trajectories.append(Trajectory(states=states, actions=actions))
    
    return trajectories


def pad_trajectory(
    trajectory: Trajectory,
    target_length: int,
    pad_state: Optional[np.ndarray] = None,
    pad_action: Optional[np.ndarray] = None
) -> Trajectory:
    """
    Pad trajectory to target length.
    
    Args:
        trajectory: Trajectory to pad
        target_length: Target length (number of states)
        pad_state: State to use for padding (default: last state)
        pad_action: Action to use for padding (default: zero action)
        
    Returns:
        Padded trajectory
    """
    current_length = len(trajectory.states)
    
    if current_length >= target_length:
        return trajectory
    
    # Get padding values
    if pad_state is None:
        pad_state = np.asarray(trajectory.states[-1], dtype=np.float32)
    if pad_action is None:
        action_dim = len(trajectory.actions[0]) if trajectory.actions else 0
        pad_action = np.zeros(action_dim, dtype=np.float32)
    
    # Pad states
    padded_states = list(trajectory.states)
    for _ in range(target_length - current_length):
        padded_states.append(pad_state.copy())
    
    # Pad actions
    padded_actions = list(trajectory.actions)
    for _ in range(target_length - 1 - len(trajectory.actions)):
        padded_actions.append(pad_action.copy())
    
    return Trajectory(states=padded_states, actions=padded_actions)


def ensure_backend_array(
    arr: Any,
    backend: str = "numpy"
) -> Union[np.ndarray, Any]:
    """
    Ensure array is in specified backend.
    
    Args:
        arr: Array in any backend
        backend: Target backend ("numpy", "jax", "torch")
        
    Returns:
        Array in target backend
    """
    if backend == "numpy":
        if isinstance(arr, np.ndarray):
            return arr
        elif JAX_AVAILABLE and hasattr(arr, '__jax_array__'):
            return np.asarray(arr)
        else:
            return np.asarray(arr)
    elif backend == "jax":
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not available")
        if hasattr(arr, '__jax_array__'):
            return arr
        else:
            return jnp.asarray(arr)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def stack_trajectories(
    trajectories: List[Trajectory],
    axis: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stack trajectories into arrays (assumes same length).
    
    Args:
        trajectories: List of trajectories (must have same length)
        axis: Axis to stack along (0 = batch dimension)
        
    Returns:
        Tuple of (states, actions) arrays
    """
    if len(trajectories) == 0:
        return np.array([]), np.array([])
    
    # Check same length
    lengths = [len(traj.states) for traj in trajectories]
    if len(set(lengths)) > 1:
        raise ValueError("All trajectories must have same length for stacking")
    
    # Stack states
    states_list = [np.stack([np.asarray(s) for s in traj.states]) for traj in trajectories]
    states = np.stack(states_list, axis=axis)
    
    # Stack actions
    actions_list = [np.stack([np.asarray(a) for a in traj.actions]) for traj in trajectories]
    actions = np.stack(actions_list, axis=axis)
    
    return states, actions


