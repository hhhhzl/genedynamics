"""
CBF convexifier backends.

Backend implementations for CBF convexifier:
- cbf_numpy: NumPy backend
- cbf_jax: JAX backend (with JIT and vmap for per-step batching)
"""

# Backend implementations will be registered here
# For now, the base CBFConvexifier handles numpy fallback

__all__ = []


