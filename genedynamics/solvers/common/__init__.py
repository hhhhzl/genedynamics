"""
Shared solver components.
"""

from importlib import import_module


def __getattr__(name):
    """Retain the legacy JAX manifold export without requiring it eagerly."""

    if name != "manifold":
        raise AttributeError(name)
    value = import_module(".manifold", __name__)
    globals()[name] = value
    return value


__all__ = ["manifold"]


