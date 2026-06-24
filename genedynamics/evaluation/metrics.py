"""General evaluation metrics — pure, reusable, task/solver-agnostic.

Every metric here is a **pure function of generic arrays** (positions, controls,
inequality residuals, forces, contact flags, ...) — never of a task object or a
particular algorithm. The same `goal_error` / `violation_cvar` / `control_smoothness`
serves corridor, stepping-stones, surface-scan, box-push, or any future task; the
*task-specific* part is only the small "signal extractor" that maps a trajectory
to these generic arrays (see `genedynamics.evaluation.plugin`).

Each metric is registered (via `@metric`) with the signal keys it consumes (read
off its signature), so callers can either:
  * call the function directly (``goal_error(final, target)``), or
  * request metrics by name from a signal dict (``compute_metrics([...], signals)``).

The by-name request is fully general — a request item may be:
  * ``"goal_error"`` — bind params to same-named signals/config; or
  * ``{"name": "violation_cvar", "as": "g_fric_cvar", "bind": {"g": "g_fric"},
       "config": {"alpha": 0.9}}`` — bind a metric to CHOSEN signal keys, give it a
    label, and per-metric config; or
  * ``("violation_cvar", "g_fric")`` — shorthand binding the first required signal.
Computed outputs are fed back into the signal namespace, so a composite metric
(``success`` off ``goal_error``/``max_violation``) composes by name. Per-metric
config keyed by label overrides the shared flat config.

Conventions:
  * inequality residual ``g``: ``g <= 0`` feasible, ``g > 0`` is a violation.
  * a "rate" is a fraction in ``[0, 1]``; a "margin" is ``>= 0`` when safe.
  * costs/violations use the HIGH tail for CVaR; rewards/margins use LOW. Metrics
    tag ``higher_is_better`` so aggregation picks the worst-case tail correctly.
  * extractors provide arrays already matching ``target`` dimensionality (they do
    any task projection); metrics do not slice columns.

NumPy only — no jax/mjx — so metrics run anywhere (offline analysis included).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

Array = Any


# ---------------------------------------------------------------------------
# Registry: each metric declares the signal keys it needs (from its signature)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Metric:
    fn: Callable[..., Any]
    required: Tuple[str, ...]    # params without a default -> mandatory signals
    optional: Tuple[str, ...]    # params with a default   -> config the caller may pass
    higher_is_better: bool       # for CVaR tail direction in aggregation


_REGISTRY: Dict[str, _Metric] = {}


def metric(fn: Callable[..., Any] | None = None, *, higher_is_better: bool = False):
    """Register a metric. Usable bare (``@metric``) or with a tag
    (``@metric(higher_is_better=True)``). Required params (no default) become the
    signals it MUST receive; optional params are config. ``*args``/``**kwargs``
    are ignored."""
    def deco(f: Callable[..., Any]) -> Callable[..., Any]:
        kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        params = inspect.signature(f).parameters.items()
        required = tuple(p for p, q in params if q.default is inspect._empty and q.kind in kinds)
        optional = tuple(p for p, q in params if q.default is not inspect._empty and q.kind in kinds)
        _REGISTRY[f.__name__] = _Metric(f, required, optional, bool(higher_is_better))
        return f
    return deco(fn) if fn is not None else deco


def get_metric(name: str) -> Callable[..., Any]:
    return _REGISTRY[name].fn


def list_metrics() -> List[str]:
    return sorted(_REGISTRY)


def metric_signals(name: str) -> Tuple[str, ...]:
    return _REGISTRY[name].required


def metric_higher_is_better(name: str) -> bool:
    return _REGISTRY[name].higher_is_better if name in _REGISTRY else False


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

@metric
def cvar(x, alpha: float = 0.95, tail: str = "high") -> float:
    """Conditional Value-at-Risk: mean of the worst ``(1-alpha)`` tail.
    ``tail='high'`` for costs/violations (worst = largest); ``'low'`` for
    rewards/margins (worst = smallest)."""
    a = _arr(x).ravel()
    if a.size == 0:
        return 0.0
    if tail == "low":
        a = -a
    # float-tolerant tail count (avoid (1-0.95)*100 == 5.0000…4 -> ceil 6)
    k = min(max(1, int(np.ceil((1.0 - alpha) * a.size - 1e-9))), a.size)
    worst = np.partition(a, a.size - k)[a.size - k:]
    m = float(np.mean(worst))
    return (-m + 0.0) if tail == "low" else m   # +0.0 normalizes -0.0


@metric
def rate(mask) -> float:
    """Fraction of True/positive entries in a boolean/0-1 array."""
    a = _arr(mask).ravel()
    return float(np.mean(a > 0.5)) if a.size else 0.0


# ---------------------------------------------------------------------------
# Task success / progress
# ---------------------------------------------------------------------------

@metric
def goal_error(final_pos, target) -> float:
    """Euclidean distance from the final position to the goal."""
    return float(np.linalg.norm(_arr(final_pos).ravel() - _arr(target).ravel()))


@metric(higher_is_better=True)
def success(goal_error, margin: float = 0.2, violation: float = 0.0,
            violation_tol: float = 1e-6) -> float:
    """1.0 if within ``margin`` of the goal AND no constraint violation. As a
    composite, ``goal_error`` (a leaf metric's output) and ``violation`` (e.g.
    bound to ``max_violation``) compose by name; if ``violation`` is unbound it
    defaults to 0 (feasibility clause inert — bind it for a real safety gate)."""
    return float(float(goal_error) <= margin and float(violation) <= violation_tol)


@metric(higher_is_better=True)
def progress_ratio(start_pos, final_pos, target) -> float:
    """Fraction of the start→goal vector covered (projected); 1.0 = reached."""
    s, f, t = _arr(start_pos).ravel(), _arr(final_pos).ravel(), _arr(target).ravel()
    total = t - s
    denom = float(total @ total)
    if denom < 1e-12:
        return 1.0
    return float(((f - s) @ total) / denom)


@metric
def completion_step(positions, target, margin: float = 0.2) -> int:
    """First timestep within ``margin`` of the goal (or T if never reached).
    ``positions`` rows must match ``target`` dimensionality (extractor projects)."""
    p = _arr(positions)
    t = _arr(target).ravel()
    d = np.linalg.norm(p - t, axis=1)
    hit = np.nonzero(d <= margin)[0]
    return int(hit[0]) if hit.size else int(p.shape[0])


@metric(higher_is_better=True)
def coverage_ratio(visited, total) -> float:
    """Fraction of target cells/waypoints/area visited."""
    return float(np.clip(float(visited) / max(float(total), 1e-12), 0.0, 1.0))


@metric(higher_is_better=True)
def path_completion(positions, reference_path) -> float:
    """Fraction of a parameterized REFERENCE path's arc-length traversed
    (project the final point onto the reference polyline) — geometry-independent,
    distinct from ``coverage_ratio``. ``[0, 1]``."""
    p = _arr(positions)
    ref = _arr(reference_path)
    if ref.shape[0] < 2 or p.shape[0] == 0:
        return 1.0 if p.shape[0] else 0.0
    seg = np.linalg.norm(ref[1:] - ref[:-1], axis=1)
    total = float(np.sum(seg))
    if total < 1e-12:
        return 1.0
    final = p[-1, : ref.shape[1]]
    best_s, best_d, acc = 0.0, np.inf, 0.0
    for i in range(ref.shape[0] - 1):
        a, b = ref[i], ref[i + 1]
        ab = b - a
        L = float(ab @ ab)
        tp = 0.0 if L < 1e-12 else float(np.clip(((final - a) @ ab) / L, 0.0, 1.0))
        proj = a + tp * ab
        d = float(np.linalg.norm(final - proj))
        if d < best_d:
            best_d, best_s = d, acc + tp * float(np.sqrt(L))
        acc += float(seg[i])
    return float(np.clip(best_s / total, 0.0, 1.0))


@metric
def pose_error(final_pose, target_pose, rot_dims=(), rot_weight: float = 1.0) -> float:
    """SE(2)/SE(3) pose error: Euclidean translation error + WRAPPED-angle
    rotation error (rad), combined with ``rot_weight``. ``rot_dims`` are the
    angle indices (wrapped); the rest are translation. Avoids the m-vs-rad mixing
    and 2π wrap bug of feeding a pose into ``goal_error``."""
    f, t = _arr(final_pose).ravel(), _arr(target_pose).ravel()
    rot = {int(i) for i in rot_dims}
    tr_idx = [i for i in range(f.shape[0]) if i not in rot]
    trans = float(np.linalg.norm(f[tr_idx] - t[tr_idx])) if tr_idx else 0.0
    rot_err = 0.0
    for i in rot:
        d = float(f[i] - t[i])
        rot_err += float(np.arctan2(np.sin(d), np.cos(d))) ** 2
    return trans + rot_weight * float(np.sqrt(rot_err))


# ---------------------------------------------------------------------------
# Constraint / safety  (g: inequality residual, >0 = violation; h: equality)
# ---------------------------------------------------------------------------

@metric
def violation_rate(g, thresh: float = 0.0) -> float:
    """Fraction of steps with an inequality violation ``g > thresh``."""
    return float(np.mean(_arr(g).ravel() > thresh)) if _arr(g).size else 0.0


@metric
def violation_mean(g) -> float:
    """Mean violation magnitude ``mean([g]_+)``."""
    a = _arr(g).ravel()
    return float(np.mean(np.maximum(a, 0.0))) if a.size else 0.0


@metric
def max_violation(g) -> float:
    """Worst (max) inequality violation ``max([g]_+)``."""
    a = _arr(g).ravel()
    return float(np.max(np.maximum(a, 0.0))) if a.size else 0.0


@metric
def violation_cvar(g, alpha: float = 0.95) -> float:
    """CVaR of the violation tail ``cvar([g]_+)`` — the headline safety metric.
    Bind ``g`` to a specific constraint (``g_fric`` / ``g_bal`` / ...) to report
    per-constraint CVaR."""
    return cvar(np.maximum(_arr(g).ravel(), 0.0), alpha=alpha, tail="high")


@metric
def equality_residual_rms(h) -> float:
    """RMS of an equality-constraint residual (``h == 0`` feasible). Bind ``h``
    to a specific residual (``h_surf`` / ``h_normal`` / ...) as needed."""
    a = _arr(h).ravel()
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


# ---------------------------------------------------------------------------
# Contact / force
# ---------------------------------------------------------------------------

@metric
def force_tracking_error(force, force_des) -> float:
    """RMS error between measured and desired (normal) force."""
    f, fd = _arr(force).ravel(), _arr(force_des).ravel()
    return float(np.sqrt(np.mean((f - fd) ** 2))) if f.size else 0.0


@metric
def force_tracking_error_tracked(force, force_des, mask) -> float:
    """RMS force error, scored ONLY over steps where ``mask`` is truthy (e.g. the EE is
    on the scan path). Force tracking is meaningful only while actually doing the task;
    a controller that sits off the path gets a trivially steady force that should not
    count. Returns NaN when there are no on-task steps (it never tracked the surface)."""
    f, fd = _arr(force).ravel(), _arr(force_des).ravel()
    m = _arr(mask).ravel().astype(bool)
    n = min(f.size, fd.size, m.size)
    if n == 0 or not m[:n].any():
        return float("nan")
    f, fd, m = f[:n], fd[:n], m[:n]
    return float(np.sqrt(np.mean((f[m] - fd[m]) ** 2)))


@metric
def force_violation_rate(force, f_min: float = 0.0, f_max: float = np.inf) -> float:
    """Fraction of steps the contact force is outside ``[f_min, f_max]``."""
    f = _arr(force).ravel()
    return float(np.mean((f < f_min) | (f > f_max))) if f.size else 0.0


@metric(higher_is_better=False)
def contact_loss_rate(in_contact) -> float:
    """Fraction of steps that LOST a contact that should hold (``in_contact``
    is 1 when contact is maintained)."""
    a = _arr(in_contact).ravel()
    return float(np.mean(a < 0.5)) if a.size else 0.0


@metric
def tangential_slip(slip_speed) -> float:
    """Mean tangential slip speed at the contact."""
    a = _arr(slip_speed).ravel()
    return float(np.mean(np.abs(a))) if a.size else 0.0


@metric
def friction_cone_violation_rate(f_tangential, f_normal, mu: float = 0.5) -> float:
    """Fraction of steps the friction cone is violated ``|f_t| > mu * f_n``.
    ``f_tangential`` is the per-step tangential MAGNITUDE (``|.|`` applied
    defensively so a signed component does not read as feasible)."""
    ft = np.abs(_arr(f_tangential).ravel())
    fn = _arr(f_normal).ravel()
    return float(np.mean(ft > mu * np.maximum(fn, 0.0))) if ft.size else 0.0


# ---------------------------------------------------------------------------
# Balance / stability
# ---------------------------------------------------------------------------

@metric
def fall_rate(fell) -> float:
    """Fraction of steps (or episodes) in a fallen state."""
    return rate(fell)


@metric(higher_is_better=True)
def balance_margin(com_xy, support_center, support_radius: float = 0.25) -> float:
    """Minimum CoM-over-support margin over time: ``radius - ||com - center||``
    (``>= 0`` balanced). Returns the worst (min) margin; empty -> 0.0 (no data)."""
    com = _arr(com_xy)
    c = _arr(support_center).ravel()
    com = com.reshape(-1, c.shape[0])
    d = np.linalg.norm(com - c, axis=1)
    return float(np.min(support_radius - d)) if d.size else 0.0


@metric(higher_is_better=True)
def tip_margin(margin_series) -> float:
    """Worst (min) tipping/safety margin over time (``>= 0`` safe);
    empty -> 0.0 (no data)."""
    a = _arr(margin_series).ravel()
    return float(np.min(a)) if a.size else 0.0


# ---------------------------------------------------------------------------
# Control quality / effort
# ---------------------------------------------------------------------------

@metric
def smoothness(sequence) -> float:
    """Mean step-to-step change ``mean ||x_{t+1} - x_t||`` of ANY sequence
    (controls, stiffness, ...). Lower is smoother. Generic; for the by-name path
    use the standard-keyed ``control_smoothness`` / ``stiffness_smoothness``."""
    a = _arr(sequence)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[0] < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(a[1:] - a[:-1], axis=1)))


@metric
def control_smoothness(controls) -> float:
    """Smoothness of the control sequence ``mean ||du||`` (signal key ``controls``)."""
    return smoothness(controls)


@metric
def stiffness_smoothness(stiffness) -> float:
    """Smoothness of the stiffness sequence ``mean ||dlogK||`` (signal key ``stiffness``)."""
    return smoothness(stiffness)


@metric
def energy(controls, scale: float = 1.0) -> float:
    """Mean control effort ``mean sum((u/scale)^2)`` (a proxy for energy)."""
    a = _arr(controls)
    if a.ndim == 1:
        a = a[:, None]
    return float(np.mean(np.sum((a / scale) ** 2, axis=1))) if a.shape[0] else 0.0


@metric
def path_length(positions) -> float:
    """Total path length ``sum ||p_{t+1} - p_t||``."""
    p = _arr(positions)
    if p.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(p[1:] - p[:-1], axis=1)))


@metric(higher_is_better=True)
def path_efficiency(positions, target) -> float:
    """Straight-line(start→target) / actual path-length, CLAMPED to ``[0, 1]``
    (1 = straight). ``positions`` must match ``target`` dimensionality."""
    p = _arr(positions)
    t = _arr(target).ravel()
    if p.shape[0] < 2:
        return 0.0
    straight = float(np.linalg.norm(t - p[0]))
    return float(np.clip(straight / max(path_length(p), 1e-9), 0.0, 1.0))


@metric
def runtime(seconds) -> float:
    """Wall-clock runtime (pass-through; for aggregation/reporting)."""
    return float(seconds)


# ---------------------------------------------------------------------------
# By-name computation from a signal dict (aliasing + chaining + per-metric cfg)
# ---------------------------------------------------------------------------

def _spec(item) -> Tuple[str, str, Dict[str, str], Dict[str, Any]]:
    """Normalize a request item -> (label, metric_name, bind, per_metric_config)."""
    if isinstance(item, str):
        return item, item, {}, {}
    if isinstance(item, dict):
        name = item["name"]
        return item.get("as", name), name, dict(item.get("bind") or {}), dict(item.get("config") or {})
    if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], str):
        name, key = item                      # ("violation_cvar", "g_fric") shorthand
        req = _REGISTRY[name].required
        return f"{name}:{key}", name, ({req[0]: key} if req else {}), {}
    raise TypeError(f"bad metric request: {item!r}")


def compute_metrics(requests: List[Any], signals: Dict[str, Any], *,
                    config: Dict[str, Any] | None = None,
                    skip_missing: bool = True) -> Dict[str, Any]:
    """Compute the requested metrics from a generic ``signals`` dict.

    Each request (str | dict | (name, key)) is resolved to (label, metric, bind,
    cfg). Required signals are pulled from the namespace (signals + already-computed
    outputs, enabling composites); optional config resolves from per-request cfg,
    then ``config[label]`` (nested), then a same-named signal, then flat ``config``.
    A metric whose required signals are absent is skipped (``skip_missing``) or
    raises.
    """
    config = config or {}
    ns: Dict[str, Any] = dict(signals)        # working namespace; outputs fed back
    out: Dict[str, Any] = {}
    for item in requests:
        label, name, bind, mcfg = _spec(item)
        m = _REGISTRY[name]
        key_for = lambda p: bind.get(p, p)
        if not all(key_for(p) in ns for p in m.required):
            if skip_missing:
                continue
            missing = [key_for(p) for p in m.required if key_for(p) not in ns]
            raise KeyError(f"metric '{label}' ({name}) missing signals {missing}")
        kw = {p: ns[key_for(p)] for p in m.required}
        per_label = config.get(label) if isinstance(config.get(label), dict) else {}
        for p in m.optional:
            if p in mcfg:
                kw[p] = mcfg[p]
            elif p in per_label:
                kw[p] = per_label[p]
            elif key_for(p) in ns:
                kw[p] = ns[key_for(p)]
            elif p in config and not isinstance(config[p], dict):
                kw[p] = config[p]
        val = m.fn(**kw)
        out[label] = val
        ns[label] = val
    return out


__all__ = [
    "metric", "get_metric", "list_metrics", "metric_signals",
    "metric_higher_is_better", "compute_metrics",
    "cvar", "rate",
    "goal_error", "success", "progress_ratio", "completion_step", "coverage_ratio",
    "path_completion", "pose_error",
    "violation_rate", "violation_mean", "max_violation", "violation_cvar",
    "equality_residual_rms",
    "force_tracking_error", "force_violation_rate", "contact_loss_rate",
    "tangential_slip", "friction_cone_violation_rate",
    "fall_rate", "balance_margin", "tip_margin",
    "smoothness", "control_smoothness", "stiffness_smoothness", "energy",
    "path_length", "path_efficiency", "runtime",
]
