import numpy as np
Array = np.ndarray


def euclidean_metric_inv(x: Array) -> Array:
    """
    return G^{-1}(x), the simplest is identity.
    x: (..., n)
    return: (n,) or (..., n, n)
    """
    n = x.shape[-1]
    return np.eye(n, dtype=x.dtype)
