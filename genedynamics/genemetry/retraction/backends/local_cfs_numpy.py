"""
NumPy local CFS retraction.

Extracts the ``_mini_batch_cfs_local`` logic from 2GO into a reusable
genemetry component.  Per-timestep within a window, pushes actions
along the SDF gradient to reduce constraint violations.

Registered as ``("retraction", "local_cfs", "numpy")``.
"""

from typing import Any, Optional

import numpy as np

from genedynamics.genemetry.base import RetractionOperator
from genedynamics.genemetry.registry import register_genemetry
from genedynamics.genemetry.types import RetractionResult


@register_genemetry("retraction", "local_cfs", "numpy")
class LocalCfsRetractionNumpy(RetractionOperator):
    """NumPy local CFS retraction.

    Parameters
    ----------
    gain : float
        CFS gradient gain.
    dt : float
        Integration timestep.
    action_limit : float
        Symmetric action clipping bound.
    """

    def __init__(
        self,
        gain: float = 0.35,
        dt: float = 0.1,
        action_limit: float = 1.0,
    ) -> None:
        self._gain = float(gain)
        self._dt = float(max(dt, 1e-6))
        self._action_limit = float(action_limit)

    def retract(
        self,
        state: Any,
        trajectory: Any,
        params: Optional[Any] = None,
    ) -> RetractionResult:
        """Apply local gradient-based CFS retraction.

        Parameters
        ----------
        state : unused.
        trajectory : (H, U) np.ndarray — actions to retract.
        params : dict with keys:
            - ``violations`` : (H,) np.ndarray — g_plus per timestep.
            - ``gradients`` : (H, dim) np.ndarray — SDF gradients.
            - ``window`` : (start, end) tuple.

        Returns
        -------
        RetractionResult with retracted actions.
        """
        params = params or {}
        violations = np.asarray(params["violations"], dtype=np.float32)
        gradients = np.asarray(params["gradients"], dtype=np.float32)
        a, b = params["window"]

        act = np.asarray(trajectory, dtype=np.float32).copy()
        U = act.shape[1]
        n_corrected = 0

        for t in range(a, b):
            if violations[t] <= 0:
                continue
            n = np.asarray(gradients[t], dtype=np.float32).reshape(-1)[:U]
            norm = float(np.linalg.norm(n))
            if norm > 1e-8:
                n = n / norm
            # Gradient-based retraction in action space.
            act[t, :U] += self._gain * (float(violations[t]) / self._dt) * n
            n_corrected += 1

        act = np.clip(act, -self._action_limit, self._action_limit)

        return RetractionResult(
            trajectory=act,
            meta={"n_corrected": n_corrected},
        )
