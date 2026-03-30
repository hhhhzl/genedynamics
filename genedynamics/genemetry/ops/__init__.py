"""
Low-level geometry operations with backend dispatch.

The module-level convenience functions look up the registered backend
and delegate.  For direct access, import from the backend module::

    from genedynamics.genemetry.ops.backends.jax_ops import (
        build_active_rows,
        project_complement_batch,
    )
"""

from genedynamics.genemetry.ops.base import GeometryOps
from genedynamics.genemetry.registry import get_genemetry_registry

# Auto-register JAX backend (if JAX is available)
try:
    from genedynamics.genemetry.ops.backends import jax_ops  # noqa: F401
except ImportError:
    pass


def get_ops(backend: str = "jax") -> GeometryOps:
    """Return the :class:`GeometryOps` instance for *backend*."""
    registry = get_genemetry_registry()
    impl_class = registry.get("ops", "geometry", backend)
    if impl_class is None:
        available = registry.list_backends("ops", "geometry")
        raise ValueError(
            f"No '{backend}' backend for geometry ops. "
            f"Available: {available}"
        )
    return impl_class()


__all__ = ["GeometryOps", "get_ops"]
