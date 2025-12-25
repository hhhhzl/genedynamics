"""
Statistical extraction utilities for constraint system.

This module provides functions to extract statistics from rollouts and
constraint evaluations, used by schedulers for adaptive parameter tuning
and by the pipeline for performance monitoring.

Performance optimizations:
- Vectorized operations: Batch processing for efficiency
- Lazy evaluation: Only compute requested statistics
- Backend-agnostic: Works with numpy, JAX, PyTorch arrays
"""

from typing import List, Optional, Dict, Any, Union
import numpy as np

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jnp = None
    JAX_AVAILABLE = False

from enerdynamics.core.types import Trajectory, State
from .types import ConstraintStats


def compute_violation(
    violations: Union[np.ndarray, List[float]],
    reduction: str = "max"
) -> float:
    """
    Compute constraint violation statistic.
    
    Args:
        violations: Array of violations (non-negative, 0 if satisfied)
        reduction: Reduction method ("max", "mean", "sum")
        
    Returns:
        Violation statistic
    """
    violations = np.asarray(violations, dtype=np.float32)
    
    if reduction == "max":
        return float(np.max(violations))
    elif reduction == "mean":
        return float(np.mean(violations))
    elif reduction == "sum":
        return float(np.sum(violations))
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def compute_min_sdf(
    sdfs: Union[np.ndarray, List[float]]
) -> float:
    """
    Compute minimum signed distance function value.
    
    Args:
        sdfs: Array of SDF values
        
    Returns:
        Minimum SDF value
    """
    sdfs = np.asarray(sdfs, dtype=np.float32)
    return float(np.min(sdfs))


def compute_feasible_rate(
    feasible_flags: Union[np.ndarray, List[bool]]
) -> float:
    """
    Compute fraction of feasible trajectories.
    
    Args:
        feasible_flags: Array of boolean feasibility flags
        
    Returns:
        Feasible rate (0.0 to 1.0)
    """
    feasible_flags = np.asarray(feasible_flags, dtype=bool)
    if len(feasible_flags) == 0:
        return 1.0
    return float(np.mean(feasible_flags))


def compute_ess(
    weights: Union[np.ndarray, List[float]],
    normalized: bool = True
) -> float:
    """
    Compute effective sample size (ESS) for importance sampling.
    
    ESS = (sum(w))^2 / sum(w^2)
    
    Args:
        weights: Array of importance weights
        normalized: Whether weights are already normalized
        
    Returns:
        Effective sample size
    """
    weights = np.asarray(weights, dtype=np.float32)
    
    if len(weights) == 0:
        return 0.0
    
    if not normalized:
        weights = weights / (np.sum(weights) + 1e-10)
    
    sum_w = np.sum(weights)
    sum_w2 = np.sum(weights ** 2)
    
    if sum_w2 < 1e-10:
        return 0.0
    
    ess = (sum_w ** 2) / sum_w2
    return float(ess)


def compute_proj_displacement(
    original: Union[np.ndarray, List[Trajectory]],
    projected: Union[np.ndarray, List[Trajectory]],
    reduction: str = "mean"
) -> float:
    """
    Compute average projection displacement.
    
    Measures how much trajectories were moved by projection operators.
    
    Args:
        original: Original trajectories (before projection)
        projected: Projected trajectories (after projection)
        reduction: Reduction method ("mean", "max", "sum")
        
    Returns:
        Average displacement
    """
    # Handle single trajectory
    if isinstance(original, Trajectory):
        original = [original]
    if isinstance(projected, Trajectory):
        projected = [projected]
    
    if len(original) != len(projected):
        raise ValueError(f"Length mismatch: {len(original)} vs {len(projected)}")
    
    displacements = []
    for orig, proj in zip(original, projected):
        # Compute displacement for each trajectory
        orig_states = np.stack([np.asarray(s) for s in orig.states])
        proj_states = np.stack([np.asarray(s) for s in proj.states])
        
        # L2 distance per state
        state_diffs = orig_states - proj_states
        state_dists = np.linalg.norm(state_diffs, axis=-1)
        
        # Average over trajectory
        traj_displacement = np.mean(state_dists)
        displacements.append(traj_displacement)
    
    displacements = np.asarray(displacements, dtype=np.float32)
    
    if reduction == "mean":
        return float(np.mean(displacements))
    elif reduction == "max":
        return float(np.max(displacements))
    elif reduction == "sum":
        return float(np.sum(displacements))
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def extract_stats(
    violations: Optional[Union[np.ndarray, List[float]]] = None,
    sdfs: Optional[Union[np.ndarray, List[float]]] = None,
    feasible_flags: Optional[Union[np.ndarray, List[bool]]] = None,
    weights: Optional[Union[np.ndarray, List[float]]] = None,
    original_trajs: Optional[List[Trajectory]] = None,
    projected_trajs: Optional[List[Trajectory]] = None,
    qp_time: float = 0.0,
    qp_count: int = 0,
) -> ConstraintStats:
    """
    Extract comprehensive constraint statistics.
    
    This is the main function for extracting statistics from rollouts.
    All arguments are optional - only requested statistics are computed.
    
    Args:
        violations: Constraint violations
        sdfs: Signed distance function values
        feasible_flags: Feasibility flags
        weights: Importance weights (for ESS computation)
        original_trajs: Original trajectories (for displacement)
        projected_trajs: Projected trajectories (for displacement)
        qp_time: Time spent in QP solving
        qp_count: Number of QP solves
        
    Returns:
        ConstraintStats object with all computed statistics
    """
    stats = ConstraintStats()
    
    # Violation
    if violations is not None:
        stats.violation = compute_violation(violations)
    
    # Minimum SDF
    if sdfs is not None:
        stats.min_sdf = compute_min_sdf(sdfs)
    
    # Feasible rate
    if feasible_flags is not None:
        stats.feasible_rate = compute_feasible_rate(feasible_flags)
    
    # ESS
    if weights is not None:
        stats.ess = compute_ess(weights)
    
    # Projection displacement
    if original_trajs is not None and projected_trajs is not None:
        stats.proj_displacement = compute_proj_displacement(
            original_trajs, projected_trajs
        )
    
    # QP statistics
    stats.qp_time = qp_time
    stats.qp_count = qp_count
    
    return stats


def extract_stats_batch(
    trajectories: List[Trajectory],
    constraint_evaluator: Any,
    position_extractor: Optional[Any] = None,
) -> ConstraintStats:
    """
    Extract statistics from batch of trajectories.
    
    This is a convenience function that evaluates constraints on a batch
    of trajectories and extracts statistics.
    
    Args:
        trajectories: List of trajectories to evaluate
        constraint_evaluator: Object with evaluate/violations methods
        position_extractor: Function to extract positions from states
        
    Returns:
        ConstraintStats object
    """
    # Evaluate constraints
    violations_list = []
    feasible_list = []
    sdfs_list = []
    
    for traj in trajectories:
        # Get violations
        if hasattr(constraint_evaluator, 'violations'):
            violations = constraint_evaluator.violations(traj)
            violations_list.extend(violations)
            
            # Check feasibility
            if hasattr(constraint_evaluator, 'is_feasible'):
                feasible = constraint_evaluator.is_feasible(traj)
                feasible_list.append(feasible)
        
        # Get SDFs if available
        if position_extractor is not None and hasattr(constraint_evaluator, 'obstacles'):
            for state in traj.states:
                pos = position_extractor(state)
                sdf = constraint_evaluator.obstacles.sdf(pos)
                sdfs_list.append(float(sdf))
    
    # Extract statistics
    return extract_stats(
        violations=violations_list if violations_list else None,
        sdfs=sdfs_list if sdfs_list else None,
        feasible_flags=feasible_list if feasible_list else None,
    )


def topK_selection(
    values: Union[np.ndarray, List[float]],
    K: int,
    largest: bool = True
) -> np.ndarray:
    """
    Select top-K values (for active constraint selection).
    
    Args:
        values: Array of values
        K: Number of top values to select
        largest: If True, select largest; if False, select smallest
        
    Returns:
        Boolean mask indicating selected values
    """
    values = np.asarray(values, dtype=np.float32)
    K = min(K, len(values))
    
    if K == 0:
        return np.zeros(len(values), dtype=bool)
    
    if largest:
        indices = np.argpartition(values, -K)[-K:]
    else:
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
    
    Alias for topK_selection (same function, different naming).
    
    Args:
        values: Array of values
        L: Number of top values to select
        largest: If True, select largest; if False, select smallest
        
    Returns:
        Boolean mask indicating selected values
    """
    return topK_selection(values, L, largest)

