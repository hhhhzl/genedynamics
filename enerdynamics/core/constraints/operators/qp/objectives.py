"""
QP objective functions.

This module provides various objective functions for QP problems:
- ||u - u_nom||^2: Nominal action tracking
- ||u_dot||^2: Jerk minimization
- Smoothness: ||u - u_prev||^2
- Weighted combinations
"""

from typing import Optional, Dict, Any
import numpy as np


def nominal_tracking_objective(
    u_nom: np.ndarray,
    weight: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Nominal tracking objective: min ||u - u_nom||^2.
    
    Args:
        u_nom: Nominal action
        weight: Weight for objective
        
    Returns:
        Tuple of (P, q) for QP: min (1/2) u^T P u + q^T u
    """
    n = len(u_nom)
    P = weight * np.eye(n, dtype=np.float32)
    q = -weight * u_nom
    return P, q


def jerk_minimization_objective(
    u_prev: Optional[np.ndarray] = None,
    dt: float = 0.1,
    weight: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Jerk minimization objective: min ||u_dot||^2.
    
    Approximates jerk as: u_dot ≈ (u - u_prev) / dt
    
    Args:
        u_prev: Previous action (if None, assumes u_prev = 0)
        dt: Time step
        weight: Weight for objective
        
    Returns:
        Tuple of (P, q) for QP
    """
    if u_prev is None:
        u_prev = np.zeros(2, dtype=np.float32)  # Default action dimension
    
    n = len(u_prev)
    P = (weight / (dt ** 2)) * np.eye(n, dtype=np.float32)
    q = -(weight / (dt ** 2)) * u_prev
    return P, q


def smoothness_objective(
    u_prev: Optional[np.ndarray] = None,
    weight: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Smoothness objective: min ||u - u_prev||^2.
    
    Encourages smooth control sequences.
    
    Args:
        u_prev: Previous action (if None, assumes u_prev = 0)
        weight: Weight for objective
        
    Returns:
        Tuple of (P, q) for QP
    """
    if u_prev is None:
        u_prev = np.zeros(2, dtype=np.float32)
    
    n = len(u_prev)
    P = weight * np.eye(n, dtype=np.float32)
    q = -weight * u_prev
    return P, q


def combined_objective(
    u_nom: np.ndarray,
    u_prev: Optional[np.ndarray] = None,
    nominal_weight: float = 1.0,
    smoothness_weight: float = 0.1,
    jerk_weight: float = 0.0,
    dt: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combined objective: weighted sum of multiple objectives.
    
    min w_nom * ||u - u_nom||^2 + w_smooth * ||u - u_prev||^2 + w_jerk * ||u_dot||^2
    
    Args:
        u_nom: Nominal action
        u_prev: Previous action
        nominal_weight: Weight for nominal tracking
        smoothness_weight: Weight for smoothness
        jerk_weight: Weight for jerk minimization
        dt: Time step
        
    Returns:
        Tuple of (P, q) for QP
    """
    n = len(u_nom)
    
    # Initialize
    P = np.zeros((n, n), dtype=np.float32)
    q = np.zeros(n, dtype=np.float32)
    
    # Nominal tracking
    if nominal_weight > 0:
        P_nom, q_nom = nominal_tracking_objective(u_nom, nominal_weight)
        P += P_nom
        q += q_nom
    
    # Smoothness
    if smoothness_weight > 0:
        P_smooth, q_smooth = smoothness_objective(u_prev, smoothness_weight)
        P += P_smooth
        q += q_smooth
    
    # Jerk
    if jerk_weight > 0:
        P_jerk, q_jerk = jerk_minimization_objective(u_prev, dt, jerk_weight)
        P += P_jerk
        q += q_jerk
    
    return P, q


def trajectory_objective(
    u_nom_trajectory: list[np.ndarray],
    u_prev: Optional[np.ndarray] = None,
    nominal_weight: float = 1.0,
    smoothness_weight: float = 0.1,
    jerk_weight: float = 0.0,
    dt: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full trajectory objective: combined objective for entire trajectory.
    
    Args:
        u_nom_trajectory: List of nominal actions
        u_prev: Previous action (for first time step)
        nominal_weight: Weight for nominal tracking
        smoothness_weight: Weight for smoothness
        jerk_weight: Weight for jerk minimization
        dt: Time step
        
    Returns:
        Tuple of (P, q) for QP over full trajectory
    """
    H = len(u_nom_trajectory)
    if H == 0:
        return np.array([]), np.array([])
    
    action_dim = len(u_nom_trajectory[0])
    n_total = H * action_dim
    
    # Initialize
    P = np.zeros((n_total, n_total), dtype=np.float32)
    q = np.zeros(n_total, dtype=np.float32)
    
    # Build objective for each time step
    for t in range(H):
        u_nom_t = u_nom_trajectory[t]
        u_prev_t = u_prev if t == 0 else u_nom_trajectory[t - 1]
        
        # Get objective for this time step
        P_t, q_t = combined_objective(
            u_nom_t, u_prev_t, nominal_weight, smoothness_weight, jerk_weight, dt
        )
        
        # Place in full trajectory matrix
        idx_start = t * action_dim
        idx_end = (t + 1) * action_dim
        P[idx_start:idx_end, idx_start:idx_end] = P_t
        q[idx_start:idx_end] = q_t
    
    return P, q

