"""
Base metric tensor functions.

Provides metric tensor computations for Riemannian optimization.
"""

import numpy as np

Array = np.ndarray


def euclidean_metric_inv(x: Array) -> Array:
    """
    Return G^{-1}(x), the simplest is identity.
    
    Args:
        x: Input array of shape (..., n)
        
    Returns:
        Inverse metric tensor of shape (n,) or (..., n, n)
    """
    n = x.shape[-1]
    return np.eye(n, dtype=x.dtype)
