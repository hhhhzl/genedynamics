"""Backend implementations for CEM."""

# Ensure JAX backend is discoverable on import
try:
    from . import cem_jax  
except ImportError:
    pass

__all__ = ["cem_jax"]

