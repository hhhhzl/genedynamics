"""Path-coverage primitives.

Generic stateful coverage tracker for "cover-the-region" tasks (cleaning,
painting, search-and-rescue, ultrasound scanning, etc.). See
:class:`CoverageMap` for the per-rollout visit-strength state that augments
the planner's carry.
"""

from genedynamics.core.coverage.coverage_map import CoverageMap

__all__ = ["CoverageMap"]
