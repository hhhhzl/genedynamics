"""
JAX manifold helpers shared by geometry-aware solvers.

.. deprecated::
    This module re-exports from ``genedynamics.genemetry.ops.backends.jax_ops``
    for backward compatibility.  New code should import from
    ``genedynamics.genemetry`` directly.
"""

from genedynamics.genemetry.ops.backends.jax_ops import (  # noqa: F401
    build_active_rows,
    project_complement_batch,
)

