"""
NumPy AGP (Augmented Geometric Projection) step.

Extracts the ``_agp_step`` logic from 2GO into a reusable genemetry
component.  The algorithm:

1. Identify active constraint times within the window (top-k by
   violation magnitude).
2. Build the constraint Jacobian A (K, H*U) and weight vector w.
3. Compute the constrained update via Woodbury matrix identity:
   ``(I + A^T W A)^{-1} u``  where  ``u = -A^T w``.
4. Inject tangent noise:  ``P xi = xi - A^T (A A^T)^{-1} A xi``.
5. Update: ``x_new = x + eta * u_woodbury + sqrt(eta) * sigma * P xi``.

Registered as ``("step", "agp", "numpy")``.
"""

from typing import Any, Tuple

import numpy as np

from genedynamics.genemetry.base import ConstrainedStep
from genedynamics.genemetry.registry import register_genemetry
from genedynamics.genemetry.types import StepResult


@register_genemetry("step", "agp", "numpy")
class AgpStepNumpy(ConstrainedStep):
    """NumPy implementation of the AGP constrained step.

    Parameters
    ----------
    active_topk : int
        Number of top-k active constraints to retain per window.
    dt : float
        Integration timestep.
    action_limit : float
        Symmetric action clipping bound.
    rng : numpy Generator, int seed, or None.
    """

    def __init__(
        self,
        active_topk: int = 8,
        dt: float = 0.1,
        action_limit: float = 1.0,
        rng: Any = None,
    ) -> None:
        self._topk = int(max(1, active_topk))
        self._dt = float(max(dt, 1e-6))
        self._action_limit = float(action_limit)
        if rng is None:
            self._rng = np.random.default_rng(0)
        elif isinstance(rng, np.random.Generator):
            self._rng = rng
        else:
            self._rng = np.random.default_rng(int(rng))

    def step(
        self,
        actions: Any,
        violations: Any,
        gradients: Any,
        window: Tuple[int, int],
        *,
        sigma: float = 0.0,
        kappa: float = 1.0,
        eta: float = 0.08,
    ) -> StepResult:
        """Compute one AGP constrained step.

        Parameters
        ----------
        actions : (H, U) np.ndarray
        violations : (H,) np.ndarray — g_plus per timestep.
        gradients : (H, dim) np.ndarray — SDF gradients.
        window : (start, end) within horizon.
        sigma, kappa, eta : overlay parameters.

        Returns
        -------
        StepResult with updated actions.
        """
        actions = np.asarray(actions, dtype=np.float32)
        violations = np.asarray(violations, dtype=np.float32)
        gradients = np.asarray(gradients, dtype=np.float32)
        H, U = actions.shape
        d = H * U
        a, b = window

        # Find active constraint timesteps in window.
        idx_local = np.argsort(violations[a:b])[::-1]
        active_rel = idx_local[: self._topk]
        active_t = [a + int(i) for i in active_rel if violations[a + int(i)] > 0]

        if not active_t:
            return StepResult(actions=actions, meta={"n_active": 0})

        K = len(active_t)

        # Build constraint Jacobian and weight vector.
        A = np.zeros((K, d), dtype=np.float32)
        w = np.zeros((K,), dtype=np.float32)
        for r, t in enumerate(active_t):
            n = -np.asarray(gradients[t], dtype=np.float32).reshape(-1)[:U]
            norm = float(np.linalg.norm(n))
            if norm > 1e-8:
                n = n / norm
            start = t * U
            A[r, start: start + U] = self._dt * n
            w[r] = float(kappa) * float(max(0.0, violations[t]))

        x = actions.reshape(-1).astype(np.float32)
        u_raw = -(A.T @ w)

        # Woodbury solve: (I + A^T diag(w) A)^{-1} u_raw
        W_inv = np.diag(1.0 / np.maximum(w, 1e-6))
        M = W_inv + (A @ A.T)
        Au = A @ u_raw
        y = np.linalg.solve(
            M + 1e-6 * np.eye(K, dtype=np.float32), Au
        )
        ginv_u = u_raw - A.T @ y

        # Tangent noise: P xi = xi - A^T (A A^T)^{-1} A xi
        xi = self._rng.normal(size=(d,)).astype(np.float32)
        AA = A @ A.T + 1e-6 * np.eye(K, dtype=np.float32)
        z = np.linalg.solve(AA, A @ xi)
        p_xi = xi - A.T @ z

        # Update.
        eta_f = float(max(eta, 1e-8))
        x_new = x + eta_f * ginv_u + np.sqrt(eta_f) * float(sigma) * p_xi
        act_new = x_new.reshape(H, U).astype(np.float32)
        act_new = np.clip(act_new, -self._action_limit, self._action_limit)

        return StepResult(
            actions=act_new,
            meta={"n_active": K, "active_timesteps": active_t},
        )
