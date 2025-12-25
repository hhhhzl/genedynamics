"""
QP operator backends.

Backend implementations for QP operators:
- per_step_filter_numpy: NumPy backend
- per_step_filter_jax: JAX backend (with JIT and batch QP)
"""

# Backend implementations will be registered here
# For now, the base PerStepQPFilter handles numpy fallback

__all__ = []

