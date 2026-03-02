"""
Backend placeholder for emerging barrier constraint scheduler.

Currently mirrors the base (fixed) behavior; can be extended for JAX later.
"""

try:
    from . import emergingbarrier_numpy  # noqa: F401
except ImportError:
    pass

