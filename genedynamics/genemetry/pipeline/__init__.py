"""
Composed geometry pipelines (probe sampling, refinement).

Auto-registers available backends on import.
"""

from genedynamics.genemetry.pipeline.probe import ProbePipeline
from genedynamics.genemetry.pipeline.refine import WindowRefinement

try:
    from genedynamics.genemetry.pipeline.backends import probe_jax  # noqa: F401
except ImportError:
    pass

try:
    from genedynamics.genemetry.pipeline.backends import refine_numpy  # noqa: F401
except ImportError:
    pass

__all__ = ["ProbePipeline", "WindowRefinement"]
