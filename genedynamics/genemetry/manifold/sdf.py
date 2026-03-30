"""
SDF constraint manifold -- orchestrator with backend dispatch.

Computes local constraint geometry from SDF-derived proxy vectors
and provides metric / tangent projections.
"""

from typing import Any

from genedynamics.genemetry.base import ConstraintManifold
from genedynamics.genemetry.types import GeometryBundle
from genedynamics.genemetry.registry import get_genemetry_registry


class SdfManifold(ConstraintManifold):
    """SDF-based constraint manifold (backend-dispatched).

    Instantiate with ``backend="jax"`` (or ``"numpy"``, etc.) and all
    calls are forwarded to the registered backend implementation.

    Parameters
    ----------
    backend : str
        Backend identifier (default ``"jax"``).
    **kwargs
        Forwarded to the backend implementation constructor.
    """

    def __init__(self, *, backend: str = "jax", **kwargs: Any) -> None:
        registry = get_genemetry_registry()
        impl_class = registry.get("manifold", "sdf", backend)
        if impl_class is None:
            available = registry.list_backends("manifold", "sdf")
            raise ValueError(
                f"No '{backend}' backend registered for SdfManifold. "
                f"Available: {available}"
            )
        self._impl: ConstraintManifold = impl_class(**kwargs)

    def geometry(
        self,
        constraint_vectors: Any,
        topk_active: int,
        eps_stab: float,
    ) -> GeometryBundle:
        return self._impl.geometry(constraint_vectors, topk_active, eps_stab)

    def project(
        self,
        vectors: Any,
        bundle: GeometryBundle,
        mode: str = "metric",
    ) -> Any:
        return self._impl.project(vectors, bundle, mode)
