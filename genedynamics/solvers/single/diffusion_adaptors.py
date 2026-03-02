"""
Diversity-aware selection adaptors for multi-mode diffusion solvers.

Provides diverse top-K selection to maintain multiple distinct modes
instead of just selecting the top-K by cost (which often collapses to a single mode).
"""

from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp


def _action_features(Y0s: jnp.ndarray, stride: int = 4) -> jnp.ndarray:
    """
    Extract diversity features from action sequences.
    
    Args:
        Y0s: (M, H, A) action sequences
        stride: Subsampling stride for feature extraction
        
    Returns:
        feats: (M, D) feature vectors
    """
    Ysub = Y0s[:, ::stride, :]  # (M, H', A)
    return Ysub.reshape((Ysub.shape[0], -1))


def _state_features(states: jnp.ndarray, stride: int = 4) -> jnp.ndarray:
    """
    Extract diversity features from state trajectories (positions).
    
    Args:
        states: (M, H+1, state_dim) state sequences
        stride: Subsampling stride for feature extraction
        
    Returns:
        feats: (M, D) feature vectors (using first 2 dims as position)
    """
    positions = states[:, ::stride, :2]  # (M, H', 2) - use first 2 dims as position
    return positions.reshape((positions.shape[0], -1))


def _pairwise_l2(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Compute L2 distance from a to each row in b.
    
    Args:
        a: (D,) single feature vector
        b: (N, D) feature matrix
        
    Returns:
        distances: (N,) L2 distances
    """
    return jnp.sqrt(jnp.sum((b - a[None, :]) ** 2, axis=-1) + 1e-9)


def greedy_diverse_select(
    feats: jnp.ndarray,
    scores: jnp.ndarray,
    C: int,
    eta: float = 1.0,
) -> jnp.ndarray:
    """
    Greedy diverse selection: select C indices maximizing score + eta * diversity.
    
    Args:
        feats: (K, D) feature vectors
        scores: (K,) scores (higher is better)
        C: number of candidates to select
        eta: diversity weight
        
    Returns:
        selected_indices: (C,) indices in [0, K)
    """
    K = feats.shape[0]
    if C >= K:
        return np.arange(K, dtype=np.int32)
    
    # Start with highest score
    first = int(np.argmax(scores))
    selected = [first]
    min_dists = _pairwise_l2(feats[first], feats)  # (K,) distances to first
    
    # Greedily add candidates
    for _ in range(C - 1):
        # Compute objective: score + eta * min_distance_to_selected
        mask = np.ones(K, dtype=bool)
        mask[selected] = False
        obj = scores + eta * min_dists
        obj = np.where(mask, obj, -np.inf)
        
        nxt = int(np.argmax(obj))
        selected.append(nxt)
        
        # Update min distances
        new_dists = _pairwise_l2(feats[nxt], feats)
        min_dists = np.minimum(min_dists, new_dists)
    
    return np.array(selected, dtype=np.int32)


def diverse_topk_modes(
    Y0s: jnp.ndarray,
    states: jnp.ndarray,
    costs: jnp.ndarray,
    C: int,
    topK_cand: int = None,
    eta: float = 1.0,
    use_state_features: bool = True,
    feature_stride: int = 4,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Select diverse top-K modes from candidate trajectories.
    
    Args:
        Y0s: (M, H, A) action sequences
        states: (M, H+1, state_dim) state sequences
        costs: (M,) costs (lower is better)
        C: number of modes to return
        topK_cand: first filter to topK_cand by cost, then do diversity (None = use all M)
        eta: diversity weight
        use_state_features: if True, use state positions for diversity; else use actions
        feature_stride: stride for feature extraction
        
    Returns:
        selected_indices: (C,) indices of selected candidates
        candidate_costs: (C,) costs of selected candidates
    """
    M = Y0s.shape[0]
    K = min(topK_cand, M) if topK_cand is not None else M
    
    # First filter: topK_cand by cost
    if K < M:
        cand_idx = np.argsort(costs)[:K]  # (K,) indices of top-K by cost
    else:
        cand_idx = np.arange(M, dtype=np.int32)
    
    Ycand = Y0s[cand_idx]  # (K, H, A)
    states_cand = states[cand_idx]  # (K, H+1, state_dim)
    costs_cand = costs[cand_idx]  # (K,)
    
    # Convert costs to scores (higher is better)
    scores_cand = -costs_cand  # (K,)
    
    # Extract features for diversity
    if use_state_features:
        feats = _state_features(states_cand, stride=feature_stride)  # (K, D)
    else:
        feats = _action_features(Ycand, stride=feature_stride)  # (K, D)
    
    # Greedy diverse selection
    sel_local = greedy_diverse_select(feats, scores_cand, C, eta)  # (C,)
    sel_global = cand_idx[sel_local]  # (C,) global indices
    
    return sel_global, costs[sel_global]
