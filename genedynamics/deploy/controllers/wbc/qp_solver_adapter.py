"""Multi-method QP solver with a candidate registry.

The WBC QP is mildly non-trivial: equality constraints (floating-base
dynamics + contact tracking), inequality constraints (friction pyramids
+ torque bounds + acceleration bounds), and a quadratic objective. No
single solver wins on every problem instance — single-support phases in
particular benefit from a more conservative method.

This adapter implements a **candidate-pool fallback chain**:

1. **Full-space OSQP** on ``min ½xᵀHx + fᵀx s.t. Cx=d, Gx≤h``.
2. **Equality projection** + **null-space reduced OSQP** on the unconstrained
   manifold (lighter problem, often warm-startable).
3. **SLSQP** on the same reduced problem (scipy fallback).
4. **trust-constr** on the reduced problem with optional repair pass.
5. **KKT active-set refinement** + **projection repair** as last resort.

Each method that finishes pushes its candidate into a registry; the best
candidate (smallest constraint violation, then smallest equality residual)
is returned. The first candidate that satisfies ``constraint_tol`` short-
circuits the chain.

This module intentionally does *not* import any high-level WBC type (no
``WBCConfig``, no ``QPProblem``); it speaks pure ``(H, f, C, d, G, h)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from genedynamics.deploy.controllers.wbc.config import SolverConfig

__all__ = ["SolveResult", "QPSolverAdapter"]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class SolveResult:
    """Outcome of a constrained-QP solve."""

    x: np.ndarray
    method: str
    eq_residual: float
    ineq_violation: float

    def is_acceptable(self, tol: float) -> bool:
        return self.eq_residual <= tol and self.ineq_violation <= tol


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class QPSolverAdapter:
    """Stateful adapter that chains QP solvers with fallback.

    Stateful only in that the full-space OSQP solver is warm-started
    between calls (the previous primal is reused as ``x0``). Construct one
    adapter per controller instance.
    """

    def __init__(self, cfg: SolverConfig) -> None:
        self.cfg = cfg
        self._osqp_warm_start: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def solve(
        self,
        H: np.ndarray,
        f: np.ndarray,
        C: np.ndarray,
        d: np.ndarray,
        G: np.ndarray,
        h: np.ndarray,
        *,
        prefer_trust_constr: bool = False,
    ) -> SolveResult:
        """Solve the WBC QP, returning the best candidate from the chain."""
        registry = _CandidateRegistry(C=C, d=d, G=G, h=h, tol=self.cfg.constraint_tol)
        total_dim = _total_dim(H, C, G)

        # 1. Full-space OSQP
        if self.cfg.use_osqp:
            full = self._solve_full_osqp(H, f, C, d, G, h)
            if full is not None and registry.register(full):
                return registry.best()

        # 2. Equality projection
        x0, N = _project_onto_equality_null_space(C, d, total_dim, self.cfg.constraint_tol)
        Hr, fr, Gr, hr = _reduce(H, f, G, h, x0, N)

        if Hr.size == 0:
            registry.register(SolveResult(x=x0, method="degenerate", eq_residual=0.0, ineq_violation=0.0))
            return registry.best()

        if Gr.size == 0:
            y = _safe_solve(Hr, -fr)
            registry.register(_make_candidate(x0 + N @ y, "reduced_unconstrained_ls", C, d, G, h))
            return registry.best()

        # 3. Reduced OSQP
        if self.cfg.use_osqp:
            y = self._solve_reduced_osqp(Hr, fr, Gr, hr)
            if y is not None and registry.register(
                _make_candidate(x0 + N @ y, "osqp_reduced", C, d, G, h)
            ):
                return registry.best()

        # 4. SLSQP
        if self.cfg.use_slsqp:
            y = _solve_slsqp(Hr, fr, Gr, hr, maxiter=self.cfg.slsqp_maxiter)
            if y is not None and registry.register(
                _make_candidate(x0 + N @ y, "slsqp", C, d, G, h)
            ):
                return registry.best()

        # 5. trust-constr (single support gets the bigger budget)
        if self.cfg.use_trust_constr:
            maxiter = (
                self.cfg.single_support_trust_constr_maxiter
                if prefer_trust_constr
                else self.cfg.trust_constr_maxiter
            )
            y = _solve_trust_constr(Hr, fr, Gr, hr, maxiter=maxiter)
            if y is not None and registry.register(
                _make_candidate(x0 + N @ y, "trust_constr", C, d, G, h)
            ):
                return registry.best()

        # 6. KKT active-set refinement
        y = _solve_active_set(
            Hr, fr, Gr, hr, refine_iters=self.cfg.active_set_refine_iters
        )
        registry.register(_make_candidate(x0 + N @ y, "active_set", C, d, G, h))

        # 7. Projection repair (if active-set still violates)
        if Gr.size > 0 and float(np.max(Gr @ y - hr)) > self.cfg.constraint_tol:
            y_proj = _projection_repair(
                y, Gr, hr,
                tol=self.cfg.constraint_tol,
                max_iters=self.cfg.projection_repair_iters,
            )
            registry.register(
                _make_candidate(x0 + N @ y_proj, "projection_repair", C, d, G, h)
            )

            # 8. trust-constr repair (final escape valve)
            if (
                self.cfg.use_trust_constr_repair
                and float(np.max(Gr @ y_proj - hr)) > self.cfg.trust_constr_repair_violation_threshold
            ):
                y_tc = _solve_trust_constr(
                    Hr, fr, Gr, hr, maxiter=self.cfg.trust_constr_repair_maxiter
                )
                if y_tc is not None:
                    registry.register(
                        _make_candidate(x0 + N @ y_tc, "trust_constr_repair", C, d, G, h)
                    )

        return registry.best()

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------

    def _solve_full_osqp(
        self,
        H: np.ndarray,
        f: np.ndarray,
        C: np.ndarray,
        d: np.ndarray,
        G: np.ndarray,
        h: np.ndarray,
    ) -> Optional[SolveResult]:
        try:
            import osqp
            from scipy import sparse
        except ImportError:
            return None
        try:
            P = sparse.csc_matrix(0.5 * (H + H.T))
            q = np.asarray(f, dtype=np.float64)
            blocks_A, lower, upper = [], [], []
            if C.size > 0:
                blocks_A.append(sparse.csc_matrix(C))
                lower.append(np.asarray(d, dtype=np.float64))
                upper.append(np.asarray(d, dtype=np.float64))
            if G.size > 0:
                blocks_A.append(sparse.csc_matrix(G))
                lower.append(np.full(h.shape, -np.inf, dtype=np.float64))
                upper.append(np.asarray(h, dtype=np.float64))
            if blocks_A:
                A = sparse.vstack(blocks_A, format="csc")
                l = np.concatenate(lower, axis=0)
                u = np.concatenate(upper, axis=0)
            else:
                A = sparse.csc_matrix((0, H.shape[0]))
                l = np.zeros((0,), dtype=np.float64)
                u = np.zeros((0,), dtype=np.float64)

            solver = osqp.OSQP()
            solver.setup(
                P=P, q=q, A=A, l=l, u=u,
                verbose=self.cfg.osqp_verbose,
                polish=self.cfg.osqp_polish,
                max_iter=self.cfg.osqp_maxiter,
                eps_abs=self.cfg.constraint_tol,
                eps_rel=self.cfg.constraint_tol,
            )
            if (
                self._osqp_warm_start is not None
                and self._osqp_warm_start.shape == q.shape
            ):
                solver.warm_start(x=self._osqp_warm_start)
            result = solver.solve()
            status = str(getattr(result.info, "status", "")).lower()
            if result.x is None or "solved" not in status:
                return None
            x = np.asarray(result.x, dtype=np.float64)
            self._osqp_warm_start = x.copy()
            return _make_candidate(x, "osqp_full", C, d, G, h)
        except Exception:
            return None

    def _solve_reduced_osqp(
        self,
        Hr: np.ndarray,
        fr: np.ndarray,
        Gr: np.ndarray,
        hr: np.ndarray,
    ) -> Optional[np.ndarray]:
        try:
            import osqp
            from scipy import sparse
        except ImportError:
            return None
        try:
            P = sparse.csc_matrix(0.5 * (Hr + Hr.T))
            q = np.asarray(fr, dtype=np.float64)
            A = sparse.csc_matrix(Gr)
            l = np.full(hr.shape, -np.inf, dtype=np.float64)
            u = np.asarray(hr, dtype=np.float64)
            solver = osqp.OSQP()
            solver.setup(
                P=P, q=q, A=A, l=l, u=u,
                verbose=self.cfg.osqp_verbose,
                polish=self.cfg.osqp_polish,
                max_iter=self.cfg.osqp_maxiter,
                eps_abs=self.cfg.constraint_tol,
                eps_rel=self.cfg.constraint_tol,
            )
            result = solver.solve()
            status = str(getattr(result.info, "status", "")).lower()
            if result.x is None or "solved" not in status:
                return None
            return np.asarray(result.x, dtype=np.float64)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Candidate registry
# ---------------------------------------------------------------------------


class _CandidateRegistry:
    """Picks the best (lowest violation, then lowest residual) candidate."""

    def __init__(
        self,
        *,
        C: np.ndarray,
        d: np.ndarray,
        G: np.ndarray,
        h: np.ndarray,
        tol: float,
    ) -> None:
        self.C, self.d, self.G, self.h = C, d, G, h
        self.tol = tol
        self._best: Optional[SolveResult] = None
        self._best_score: Optional[Tuple[float, float]] = None

    def register(self, candidate: SolveResult) -> bool:
        """Register a candidate. Returns ``True`` if it is acceptable."""
        if not np.all(np.isfinite(candidate.x)):
            return False
        score = (candidate.ineq_violation, candidate.eq_residual)
        if self._best is None or score < self._best_score:  # type: ignore[operator]
            self._best = candidate
            self._best_score = score
        return candidate.is_acceptable(self.tol)

    def best(self) -> SolveResult:
        if self._best is None:
            raise RuntimeError("No QP candidate succeeded; even the fallbacks failed")
        return self._best


# ---------------------------------------------------------------------------
# Free helpers
# ---------------------------------------------------------------------------


def _total_dim(H: np.ndarray, C: np.ndarray, G: np.ndarray) -> int:
    if H.size:
        return int(H.shape[1])
    if C.size:
        return int(C.shape[1])
    if G.size:
        return int(G.shape[1])
    raise ValueError("All of H, C, G are empty; cannot infer problem dimension")


def _project_onto_equality_null_space(
    C: np.ndarray, d: np.ndarray, total_dim: int, reg: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(x0, N)`` such that any ``x = x0 + N·y`` satisfies ``C·x = d``."""
    if C.size == 0:
        return np.zeros((total_dim,), dtype=np.float64), np.eye(total_dim, dtype=np.float64)
    CCt = C @ C.T
    x0 = C.T @ _safe_solve(
        CCt + max(reg, 1e-12) * np.eye(CCt.shape[0], dtype=np.float64), d
    )
    _, S, Vt = np.linalg.svd(C, full_matrices=True)
    rank = int(np.sum(S > 1e-8))
    N = Vt[rank:].T
    return x0, N


def _reduce(
    H: np.ndarray,
    f: np.ndarray,
    G: np.ndarray,
    h: np.ndarray,
    x0: np.ndarray,
    N: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Hr = N.T @ H @ N
    fr = N.T @ (H @ x0 + f)
    if G.size:
        Gr = G @ N
        hr = h - G @ x0
    else:
        Gr = np.zeros((0, N.shape[1]), dtype=np.float64)
        hr = np.zeros((0,), dtype=np.float64)
    return Hr, fr, Gr, hr


def _make_candidate(
    x: np.ndarray,
    method: str,
    C: np.ndarray,
    d: np.ndarray,
    G: np.ndarray,
    h: np.ndarray,
) -> SolveResult:
    eq = float(np.linalg.norm(C @ x - d)) if C.size else 0.0
    ineq = float(np.max(np.maximum(G @ x - h, 0.0))) if G.size else 0.0
    return SolveResult(x=np.asarray(x, dtype=np.float64), method=method,
                       eq_residual=eq, ineq_violation=ineq)


def _safe_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        return x


def _solve_slsqp(Hr, fr, Gr, hr, *, maxiter: int) -> Optional[np.ndarray]:
    try:
        from scipy.optimize import minimize

        objective: Callable[[np.ndarray], float] = lambda y: 0.5 * float(y @ (Hr @ y)) + float(fr @ y)
        gradient: Callable[[np.ndarray], np.ndarray] = lambda y: Hr @ y + fr
        constraints = [
            {
                "type": "ineq",
                "fun": lambda y, G=Gr, h=hr: h - G @ y,
                "jac": lambda y, G=Gr, h=hr: -G,
            }
        ]
        y0 = np.zeros((Hr.shape[0],), dtype=np.float64)
        result = minimize(
            objective,
            y0,
            jac=gradient,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": int(maxiter), "ftol": 1e-8, "disp": False},
        )
        if result.success and np.all(np.isfinite(result.x)):
            return np.asarray(result.x, dtype=np.float64)
    except Exception:
        pass
    return None


def _solve_trust_constr(Hr, fr, Gr, hr, *, maxiter: int) -> Optional[np.ndarray]:
    try:
        from scipy.optimize import LinearConstraint, minimize

        objective = lambda y: 0.5 * float(y @ (Hr @ y)) + float(fr @ y)
        gradient = lambda y: Hr @ y + fr
        constraints = [LinearConstraint(Gr, -np.inf * np.ones_like(hr), hr)]
        y0 = np.zeros((Hr.shape[0],), dtype=np.float64)
        result = minimize(
            objective,
            y0,
            jac=gradient,
            method="trust-constr",
            constraints=constraints,
            options={"maxiter": int(maxiter), "verbose": 0},
        )
        if result.success and np.all(np.isfinite(result.x)):
            return np.asarray(result.x, dtype=np.float64)
    except Exception:
        pass
    return None


def _solve_active_set(
    Hr: np.ndarray,
    fr: np.ndarray,
    Gr: np.ndarray,
    hr: np.ndarray,
    *,
    refine_iters: int,
) -> np.ndarray:
    """Newton + active-set refinement starting from the unconstrained min."""
    y = _safe_solve(Hr, -fr)
    if Gr.size == 0:
        return y
    violation = Gr @ y - hr
    for _ in range(int(refine_iters)):
        active = violation > 1e-6
        if not np.any(active):
            break
        G_a = Gr[active]
        h_a = hr[active]
        KKT = np.block(
            [
                [Hr, G_a.T],
                [G_a, np.zeros((G_a.shape[0], G_a.shape[0]), dtype=np.float64)],
            ]
        )
        rhs = np.concatenate([-fr, h_a], axis=0)
        sol = _safe_solve(KKT, rhs)
        y = sol[: Hr.shape[0]]
        violation = Gr @ y - hr
    return y


def _projection_repair(
    y: np.ndarray,
    Gr: np.ndarray,
    hr: np.ndarray,
    *,
    tol: float,
    max_iters: int,
) -> np.ndarray:
    """Per-row Frank-Wolfe-style projection onto each violated half-space."""
    y_proj = y.copy()
    for _ in range(int(max_iters)):
        max_violation = 0.0
        for i in range(Gr.shape[0]):
            gi = Gr[i]
            viol = float(gi @ y_proj - hr[i])
            if viol <= tol:
                continue
            denom = float(gi @ gi) + 1e-12
            y_proj = y_proj - (viol / denom) * gi
            if viol > max_violation:
                max_violation = viol
        if max_violation <= tol:
            break
    return y_proj
