"""General aggregation of per-run metric dicts (over seeds / levels / methods).

Task- and solver-agnostic: takes a list of ``{metric_name: value}`` records and
reports, per numeric metric, ``mean / std / min / max / cvar95`` — the same
summary the experiment runner produces, but as a reusable function any caller can
apply to any metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from genedynamics.evaluation.metrics import cvar, metric_higher_is_better


def _tail_for(key: str, higher_is_better: set) -> str:
    """CVaR tail: 'low' (worst = smallest) for higher-is-better metrics
    (success, margins, efficiency, coverage, ...); 'high' otherwise (costs,
    violations). Resolved from the metric registry (stripping any ``name:alias``
    binding suffix) plus an explicit override set."""
    base = key.split(":", 1)[0]
    return "low" if (key in higher_is_better or base in higher_is_better
                     or metric_higher_is_better(base)) else "high"


def aggregate(records: List[Dict[str, Any]], *, alpha: float = 0.95,
              higher_is_better: set | None = None) -> Dict[str, Dict[str, float]]:
    """Per-metric ``{mean,std,min,max,cvar95}`` over a list of run records.

    Non-numeric values are ignored; ``bool`` -> 0/1 (so ``success`` aggregates
    into a success RATE). The CVaR tail is chosen PER METRIC: higher-is-better
    metrics use the LOW tail (worst case = smallest), costs/violations use HIGH —
    so cvar95 is always the worst-case. ``higher_is_better`` adds custom labels
    (e.g. aliased metrics) the registry can't know about."""
    hib = set(higher_is_better or set())
    keys = sorted({k for r in records for k in r})
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = []
        for r in records:
            v = r.get(k)
            if isinstance(v, bool):
                vals.append(float(v))
            elif isinstance(v, (int, float, np.integer, np.floating)):
                vals.append(float(v))
        if not vals:
            continue
        a = np.asarray(vals, dtype=np.float64)
        out[k] = {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "cvar95": cvar(a, alpha=alpha, tail=_tail_for(k, hib)),
            "n": int(a.size),
        }
    return out


def aggregate_by(records: List[Dict[str, Any]], group_key: str, *,
                 alpha: float = 0.95,
                 higher_is_better: set | None = None) -> Dict[Any, Dict[str, Dict[str, float]]]:
    """Group records by ``record[group_key]`` (e.g. obstacle level, method) then
    :func:`aggregate` within each group."""
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(r.get(group_key), []).append(r)
    return {g: aggregate(rs, alpha=alpha, higher_is_better=higher_is_better)
            for g, rs in groups.items()}


__all__ = ["aggregate", "aggregate_by"]
