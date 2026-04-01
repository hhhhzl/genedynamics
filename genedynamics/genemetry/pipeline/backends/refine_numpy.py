"""
NumPy window-by-window refinement pipeline.

Extracts the ``_refine_candidates`` logic from 2GO into a reusable
genemetry component.  For each sliding window:

1. Compute per-window gating via multimodality proxy.
2. Identify tail candidates by total violation.
3. For each tail candidate exceeding the CVaR threshold:
   a. Run an AGP constrained step.
   b. Run local CFS retraction.
   c. Accept if cost or total violation improves.

Registered as ``("pipeline", "refine", "numpy")``.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from genedynamics.genemetry.base import RefinementPipeline
from genedynamics.genemetry.registry import register_genemetry
from genedynamics.genemetry.types import RefinementResult


def _cvar(x: np.ndarray, alpha: float) -> float:
    """Conditional Value-at-Risk: mean of the top (1-alpha) fraction."""
    arr = np.asarray(x, dtype=np.float32).ravel()
    if arr.size == 0:
        return 0.0
    n_tail = max(1, int(np.ceil((1.0 - alpha) * arr.size)))
    part = np.partition(arr, arr.size - n_tail)[arr.size - n_tail:]
    return float(np.mean(part))


@register_genemetry("pipeline", "refine", "numpy")
class WindowRefinementNumpy(RefinementPipeline):
    """NumPy window-by-window refinement pipeline.

    Parameters
    ----------
    window_policy : WindowPolicy
        Provides ``slices(horizon)`` for window iteration.
    constrained_step : ConstrainedStep
        AGP step implementation.
    local_retraction : RetractionOperator
        Local CFS retraction implementation.
    multimodality_evaluator : optional
        Window-level multimodality proxy.  If ``None``, gating is skipped
        (all windows are active).
    cvar_alpha : float
        CVaR confidence level.
    tail_ratio : float
        Fraction of candidates treated as tail (worst).
    enable_sample_tail : bool
        Restrict refinement to tail candidates only.
    enable_local_gating : bool
        Apply window-level gating.
    """

    def __init__(
        self,
        window_policy: Any = None,
        constrained_step: Any = None,
        local_retraction: Any = None,
        multimodality_evaluator: Any = None,
        cvar_alpha: float = 0.9,
        tail_ratio: float = 0.3,
        enable_sample_tail: bool = True,
        enable_local_gating: bool = True,
        rollout_fn: Optional[Callable] = None,
    ) -> None:
        self._window_policy = window_policy
        self._step = constrained_step
        self._local_retract = local_retraction
        self._multimodality = multimodality_evaluator
        self._cvar_alpha = float(cvar_alpha)
        self._tail_ratio = float(tail_ratio)
        self._enable_sample_tail = bool(enable_sample_tail)
        self._enable_local_gating = bool(enable_local_gating)
        self._rollout_fn = rollout_fn

    def refine(
        self,
        candidate_actions: List[Any],
        candidate_states: List[Any],
        violation_fn: Callable[[Any, float], Tuple[Any, Any]],
        cost_fn: Callable[[Any], float],
        schedule_params: Dict[str, Any],
        *,
        clearance: float = 0.05,
    ) -> RefinementResult:
        """Refine candidates via window-local AGP + local CFS.

        Parameters
        ----------
        candidate_actions : list of (H, U) np.ndarray
        candidate_states : list of (H+1, state_dim) np.ndarray
        violation_fn : ``(states, clearance) -> (g_plus, grad)``
        cost_fn : ``(actions) -> float``
        schedule_params : dict with ``sigma_hist``, ``delta_hist``,
            ``theta_hist``, ``eta_hist``, ``kappa_hist`` arrays of
            length K (diffusion steps).
        clearance : float

        Returns
        -------
        RefinementResult
        """
        cand_actions = [np.asarray(a, dtype=np.float32) for a in candidate_actions]
        cand_states = [np.asarray(s, dtype=np.float32) for s in candidate_states]
        C = len(cand_actions)
        H = int(cand_actions[0].shape[0])

        # rollout_fn can be overridden per-call via schedule_params
        # (useful when x0 varies between calls).
        rollout_fn = schedule_params.get("rollout_fn", self._rollout_fn)

        sigma_hist = np.asarray(schedule_params.get("sigma_hist", [0.0]), dtype=np.float32).ravel()
        delta_hist = np.asarray(schedule_params.get("delta_hist", [1e3]), dtype=np.float32).ravel()
        theta_hist = np.asarray(schedule_params.get("theta_hist", [0.5]), dtype=np.float32).ravel()
        eta_hist = np.asarray(schedule_params.get("eta_hist", [0.08]), dtype=np.float32).ravel()
        kappa_hist = np.asarray(schedule_params.get("kappa_hist", [1.0]), dtype=np.float32).ravel()
        K = max(1, sigma_hist.shape[0])

        windows = self._window_policy.slices(H) if self._window_policy is not None else [(0, H)]

        # Compute per-candidate violations.
        traces: List[Tuple[np.ndarray, np.ndarray]] = []
        viol_tot = []
        for s in cand_states:
            g, grad = violation_fn(s, clearance)
            traces.append((g, grad))
            viol_tot.append(float(np.sum(g)))
        viol_tot_arr = np.asarray(viol_tot, dtype=np.float32)

        # Identify tail candidates.
        if self._enable_sample_tail:
            n_tail = max(1, int(np.ceil(self._tail_ratio * C)))
            tail_idx = np.argsort(viol_tot_arr)[::-1][:n_tail]
        else:
            tail_idx = np.arange(C, dtype=np.int32)

        # Pre-compute candidate costs.
        cand_costs = np.asarray([cost_fn(a) for a in cand_actions], dtype=np.float32)

        window_gamma: List[float] = []
        window_cvar: List[float] = []
        window_delta: List[float] = []
        refined_indices: List[int] = []

        for wi, w in enumerate(windows):
            a, b = w
            # Map window position to schedule index.
            progress = float((b - 1) / max(1, H - 1))
            sched_idx = int(np.clip(round(progress * (K - 1)), 0, K - 1))
            sigma_w = float(sigma_hist[sched_idx])
            delta_w = float(delta_hist[sched_idx])
            kappa_w = float(kappa_hist[sched_idx])
            theta_w = float(theta_hist[sched_idx])
            eta_w = float(eta_hist[sched_idx])

            # Window-level gating.
            if self._enable_local_gating and self._multimodality is not None:
                pi_multi = self._multimodality.evaluate(cand_states, a, b)
            else:
                pi_multi = 1.0
            gamma_w = 1.0 if pi_multi > theta_w else 0.0

            window_gamma.append(gamma_w)
            window_delta.append(delta_w)

            if gamma_w <= 0:
                window_cvar.append(0.0)
                continue

            # Refine tail candidates within this window.
            cvars = []
            for ci in tail_idx:
                g, grad = traces[int(ci)]
                cvar_w = _cvar(g[a:b], self._cvar_alpha)
                cvars.append(cvar_w)
                if cvar_w <= delta_w:
                    continue

                # AGP step.
                act0 = cand_actions[int(ci)]
                if self._step is not None:
                    step_result = self._step.step(
                        act0, g, grad, w,
                        sigma=sigma_w, kappa=kappa_w, eta=eta_w,
                    )
                    act1 = step_result.actions
                else:
                    act1 = act0

                # Local CFS retraction.
                if self._local_retract is not None:
                    retract_result = self._local_retract.retract(
                        None, act1,
                        {"violations": g, "gradients": grad, "window": w},
                    )
                    act2 = np.asarray(retract_result.trajectory, dtype=np.float32)
                else:
                    act2 = act1

                # Re-rollout and evaluate.
                cost2 = cost_fn(act2)
                if rollout_fn is not None:
                    st2 = rollout_fn(act2)
                    g2, grad2 = violation_fn(st2, clearance)
                else:
                    st2 = cand_states[int(ci)]
                    g2, grad2 = g, grad

                # Accept if improves cost or reduces total violation.
                old_cost = float(cand_costs[int(ci)])
                if (cost2 <= old_cost) or (np.sum(g2) < np.sum(g)):
                    cand_actions[int(ci)] = act2
                    cand_states[int(ci)] = np.asarray(st2, dtype=np.float32)
                    cand_costs[int(ci)] = cost2
                    traces[int(ci)] = (g2, grad2)
                    refined_indices.append(int(ci))

            window_cvar.append(float(np.mean(cvars)) if cvars else 0.0)

        # Recompute costs after all refinements.
        cand_costs = np.asarray([cost_fn(a) for a in cand_actions], dtype=np.float32)
        best_idx = int(np.argmin(cand_costs))

        return RefinementResult(
            candidate_actions=cand_actions,
            candidate_states=cand_states,
            candidate_costs=cand_costs,
            best_idx=best_idx,
            window_gamma_hist=np.asarray(window_gamma, dtype=np.float32),
            window_cvar_hist=np.asarray(window_cvar, dtype=np.float32),
            window_delta_hist=np.asarray(window_delta, dtype=np.float32),
            refined_indices=np.asarray(sorted(set(refined_indices)), dtype=np.int32),
        )
