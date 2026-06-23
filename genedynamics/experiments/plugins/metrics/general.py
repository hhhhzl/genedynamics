"""Framework adapter for the general evaluation-metrics library.

`GeneralMetricsPlugin` wraps the task-agnostic `genedynamics.evaluation` metrics
into the experiment `MetricsPlugin` interface. The plugin itself is GENERAL: you
give it a list of general metric names + a small task ``extractor`` that maps a
``Trajectory``/``env`` to the generic signals the metrics consume (positions,
controls, inequality residuals ``g``, forces, contact flags, ...). All the metric
math lives in the shared library — nothing here is task- or algorithm-specific.

Example::

    def arm_signals(traj, env, obstacles, constraints, **kw):
        states = np.stack([np.ravel(s) for s in traj.states])
        return {
            "positions": states[:, :3],
            "final_pos": states[-1, :3],
            "target": env.target,
            "controls": np.stack([np.ravel(a) for a in traj.actions]),
            "g": kw.get("violations", np.zeros(len(states))),
        }

    plugin = GeneralMetricsPlugin(
        ["success", "goal_error", "violation_cvar", "smoothness", "energy"],
        extractor=arm_signals, name="arm_metrics",
    )

This is how MDAC's surface-scan / box-push experiments (and corridor, stepping,
...) all report the SAME general metrics — only their extractor differs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from genedynamics.core.types import Trajectory
from genedynamics.evaluation.metrics import compute_metrics
from genedynamics.experiments.framework.base import MetricsPlugin

Extractor = Callable[..., Dict[str, Any]]


class GeneralMetricsPlugin(MetricsPlugin):
    """Compute named general metrics via a task-provided signal extractor."""

    def __init__(self, metric_names: List[str], extractor: Extractor, *,
                 name: str = "metrics", config: Optional[Dict[str, Any]] = None,
                 skip_missing: bool = True) -> None:
        self._names = list(metric_names)
        self._extractor = extractor
        self._name = str(name)
        self._config = dict(config or {})
        self._skip_missing = bool(skip_missing)

    @property
    def name(self) -> str:
        return self._name

    def compute(self, trajectory: Trajectory, env: Any, obstacles: Any,
                constraints: Any, **kwargs: Any) -> Dict[str, Any]:
        signals = self._extractor(trajectory, env, obstacles, constraints, **kwargs)
        config = {**self._config, **kwargs}
        return compute_metrics(self._names, signals, config=config,
                               skip_missing=self._skip_missing)


__all__ = ["GeneralMetricsPlugin"]
