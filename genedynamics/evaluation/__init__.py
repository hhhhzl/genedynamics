"""General evaluation metrics + aggregation (task- and solver-agnostic).

A reusable library of pure-array performance/safety metrics that ANY task or
solver can call — directly (``goal_error(final, target)``), by name from a
signal dict (``compute_metrics([...], signals)``), or through the experiment
framework adapter (``experiments.plugins.metrics.general.GeneralMetricsPlugin``).
NumPy only; no jax/mjx/experiment dependency, so it runs in offline analysis too.
"""

from genedynamics.evaluation.metrics import (
    compute_metrics,
    get_metric,
    list_metrics,
    metric,
    metric_signals,
)
from genedynamics.evaluation.aggregate import aggregate, aggregate_by

__all__ = [
    "compute_metrics", "get_metric", "list_metrics", "metric", "metric_signals",
    "aggregate", "aggregate_by",
]
