"""Backend implementations for CEM."""

# Ensure JAX backend is discoverable on import
try:
    from . import cem_jax  # noqa: F401
except ImportError:
    pass

__all__ = ["cem_jax"]

