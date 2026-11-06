from .twogo_jax import TwoGOBackendJax

try:
    from .twogo_torch import TwoGOBackendTorch
except ImportError:
    TwoGOBackendTorch = None

__all__ = ["TwoGOBackendJax", "TwoGOBackendTorch"]

