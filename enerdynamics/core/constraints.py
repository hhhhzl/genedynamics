import numpy as np
Array = np.ndarray


def project_box(x: Array, lower: Array, upper: Array) -> Array:
    """project to box constraint [lower, upper]"""
    return np.minimum(np.maximum(x, lower), upper)


def soft_box_energy(x: Array, lower: Array, upper: Array, alpha: float = 10.0) -> float:
    """soft energy for box constraint: linear/square penalty if out of bounds"""
    below = np.maximum(0.0, lower - x)
    above = np.maximum(0.0, x - upper)
    return float(alpha * (np.sum(below ** 2) + np.sum(above ** 2)))
