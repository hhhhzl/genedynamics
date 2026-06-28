"""Backend implementations for ISSA."""

try:
    from . import issa_jax  
except ImportError:
    pass

__all__ = ["issa_jax"]
