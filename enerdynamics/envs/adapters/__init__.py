"""
Environment adapters for third-party libraries.

This module provides adapters to integrate third-party environment libraries
(Gymnasium, Brax, etc.) with the enerdynamics framework, allowing:
- Unified interface across different environment libraries
- Obstacle integration
- Multi-backend support
"""

from enerdynamics.envs.adapters.gymnasium_adapter import GymnasiumEnvAdapter
from enerdynamics.envs.adapters.unified_adapter import UnifiedEnvAdapter

__all__ = [
    "GymnasiumEnvAdapter",
    "UnifiedEnvAdapter",
]

# Brax adapter is optional (requires Brax and JAX)
try:
    import brax
    import jax
    from enerdynamics.envs.adapters.brax_adapter import BraxEnvAdapter
    __all__.append("BraxEnvAdapter")
except (ImportError, AttributeError):
    # AttributeError can occur if jax is None when accessing jax.Array in type hints
    pass
