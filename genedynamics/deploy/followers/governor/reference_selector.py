"""Revised reference-governor framework, Steps 1, 3, 4.

The deploy-side realisation of the corrected theory (see the governor docs):

* **Step 1** — feasible-AND-trackable reference selection. A reference must lie
  in ``R_adm ∩ T_policy``: geometrically feasible (body-SDF clearance) AND
  trackable by the policy (demanded lateral velocity / yaw-rate within the
  policy command envelope). We prune modes outside ``T_policy`` with a cheap
  kinematic proxy, then pick the survivor with the best *executed* clearance
  (planned clearance alone is misleading — high-clearance modes are often
  untrackable). :func:`select_mode`.
* **Step 3** — local ISS: the policy is assumed ISS only on ``T_policy``. The
  tracking error of the *selected* (trackable) mode is measured empirically.
  :func:`measure_epsilon_track`.
* **Step 4** — tighten ``m_track = L_g · ε_max`` (SDF is 1-Lipschitz ⇒ L_g=1)
  and emit an a-posteriori safety certificate ``g(x) ≤ 0``.
  :func:`m_track_from_epsilon`, :func:`certify_safety`.

Pure NumPy. The (expensive) executed rollout is injected as a callable so this
module has no spark / mujoco / docker dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PolicyEnvelope:
    """The tracking policy's *trackable* command envelope (measured, T_policy).

    Demanded references beyond this are not ISS-trackable (the policy drifts);
    they are pruned in Step 1. Defaults are the G1 spark policy's calibrated
    envelope (a touch above the training ranges, where it still tracks).
    """

    v_max_lon: float = 1.0
    v_max_lat: float = 0.55
    yaw_rate_max: float = 0.65
    # generous prune factor: only reject modes that clearly exceed the envelope
    prune_factor: float = 1.15


@dataclass
class ModeReport:
    idx: int
    planned_clear: float
    demand_v_lat: float
    demand_yaw_rate: float
    trackable: bool
    executed_clear: Optional[float] = None
    fell: Optional[bool] = None


# ---------------------------------------------------------------------------
# kinematics of a planned mode (state layout: [x, y, psi, ...]; cols 0,1,2)
# ---------------------------------------------------------------------------
def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def demanded_kinematics(states: np.ndarray, dt: float) -> Tuple[float, float]:
    """Max demanded |body-frame lateral velocity| and |yaw rate| of a plan.

    Uses only (x, y, psi) = cols (0, 1, 2) via finite differences, so it is
    robust to the rest of the state layout.
    """
    s = np.asarray(states, dtype=float)
    if s.shape[0] < 2:
        return 0.0, 0.0
    xy = s[:, :2]
    psi = s[:, 2]
    v_world = np.diff(xy, axis=0) / dt                      # (T-1, 2) world vel
    psi_mid = psi[:-1]
    c, sn = np.cos(-psi_mid), np.sin(-psi_mid)
    v_lat = sn * v_world[:, 0] + c * v_world[:, 1]          # body-frame lateral
    yaw_rate = _wrap(np.diff(psi)) / dt
    return float(np.max(np.abs(v_lat))), float(np.max(np.abs(yaw_rate)))


def select_mode(
    candidate_states: List[Any],
    planned_clearance_fn: Callable[[np.ndarray], float],
    dt: float,
    envelope: PolicyEnvelope,
    executed_eval: Optional[Callable[[int], Tuple[float, bool]]] = None,
    feasible_clear: float = 0.0,
) -> Tuple[int, List[ModeReport]]:
    """Step 1: pick the best mode in ``R_adm ∩ T_policy``.

    Parameters
    ----------
    candidate_states : list of (T, D) planned states, one per mode.
    planned_clearance_fn : (T, D) states -> min planned body-SDF clearance.
    dt : plan step (s) for the kinematic proxy.
    envelope : the policy's trackable command envelope (T_policy proxy).
    executed_eval : optional ``idx -> (executed_min_clearance, fell)``. When
        given, survivors are ranked by *executed* clearance (ground truth);
        otherwise by planned clearance.
    feasible_clear : a mode counts as geometrically feasible if its planned
        clearance ≥ this (default 0).

    Returns ``(best_idx, per-mode reports)``. Falls back to the max-planned-
    clearance mode if no trackable+feasible+executed-safe mode exists.
    """
    reports: List[ModeReport] = []
    vlat_cap = envelope.v_max_lat * envelope.prune_factor
    yaw_cap = envelope.yaw_rate_max * envelope.prune_factor
    for i, c in enumerate(candidate_states):
        S = np.asarray(c, dtype=float)
        pc = float(planned_clearance_fn(S))
        dvl, dyr = demanded_kinematics(S, dt)
        trackable = (dvl <= vlat_cap) and (dyr <= yaw_cap)
        reports.append(ModeReport(i, pc, dvl, dyr, trackable))

    # candidate pool: trackable AND geometrically feasible
    pool = [r for r in reports if r.trackable and r.planned_clear >= feasible_clear]
    if not pool:  # nothing trackable+feasible -> least-bad by planned clearance
        best = max(reports, key=lambda r: r.planned_clear)
        return best.idx, reports

    if executed_eval is not None:
        for r in pool:
            ec, fell = executed_eval(r.idx)
            r.executed_clear, r.fell = float(ec), bool(fell)
        safe = [r for r in pool if r.executed_clear is not None and r.executed_clear >= 0.0 and not r.fell]
        ranked = safe if safe else pool
        best = max(ranked, key=lambda r: (r.executed_clear if r.executed_clear is not None else -1e9))
    else:
        best = max(pool, key=lambda r: r.planned_clear)
    return best.idx, reports


# ---------------------------------------------------------------------------
# Step 3 — local-ISS tracking-error measurement on the selected (trackable) mode
# ---------------------------------------------------------------------------
def measure_epsilon_track(executed_xy: np.ndarray, planned_xy: np.ndarray) -> float:
    """LOOSE (pessimistic) positional tracking error ``max_t ‖x_t − r^g_t‖``.

    Kept for reference, but do NOT use this for Step-4 tightening: it is
    dominated by *longitudinal* lag (the robot trails the plan along the
    corridor), which does not degrade obstacle clearance. The Lipschitz bound
    g(x) ≥ g(r) − ‖x−r‖ with this ε is far too conservative. Use
    :func:`epsilon_from_clearance` instead.
    """
    x = np.asarray(executed_xy, dtype=float)
    r = np.asarray(planned_xy, dtype=float)
    n = min(len(x), len(r))
    if n == 0:
        return 0.0
    return float(np.max(np.linalg.norm(x[:n] - r[:n], axis=1)))


def epsilon_from_clearance(planned_min_clear: float, executed_min_clear: float) -> float:
    """Safety-relevant tracking error = body-SDF clearance degradation.

    ``ε = [planned_clear − executed_clear]^+`` = the actual loss of body-SDF
    margin from plan to execution. This is the directional (SDF-gradient ≈
    lateral) error that Step 4 needs: with planned ``g(r) ≤ −m_track`` and
    ``g(x) − g(r) = planned_clear − executed_clear ≤ ε``, choosing
    ``m_track ≥ ε`` guarantees the executed ``g(x) ≤ 0``. Empirically ε≈1 cm,
    vs the ~0.4 m positional error which is mostly harmless longitudinal lag.
    """
    return float(max(0.0, float(planned_min_clear) - float(executed_min_clear)))


# ---------------------------------------------------------------------------
# Step 4 — tightening + a-posteriori certificate
# ---------------------------------------------------------------------------
def m_track_from_epsilon(eps_track: float, L_g: float = 1.0, safety_factor: float = 1.3) -> float:
    """m_track = L_g · ε_max · safety_factor (SDF ⇒ L_g = 1)."""
    return float(L_g) * float(eps_track) * float(safety_factor)


def certify_safety(executed_min_clearance: float, tol: float = 0.0) -> bool:
    """A-posteriori certificate: executed body-SDF clearance ≥ -tol ⇒ g(x) ≤ 0."""
    return bool(executed_min_clearance >= -float(tol))
