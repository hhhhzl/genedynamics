"""
NumPy window-level multimodality proxy.

Computes the normalized spread of candidate trajectory endpoints within
a window.  Used by the refinement pipeline to decide whether geometric
corrections are needed at the window level.

Registered as ``("window", "multimodality", "numpy")``.
"""

import numpy as np

from genedynamics.genemetry.registry import register_genemetry


@register_genemetry("window", "multimodality", "numpy")
class WindowMultimodalityNumpy:
    """NumPy implementation of window-level multimodality proxy."""

    def __init__(self, multi_scale: float = 0.25, position_dim: int = 2) -> None:
        self._multi_scale = max(float(multi_scale), 1e-6)
        self._position_dim = int(position_dim)

    def evaluate(
        self,
        candidate_states: list,
        window_start: int,
        window_end: int,
    ) -> float:
        """Compute normalized spread of window-end positions.

        Parameters
        ----------
        candidate_states : list of (T, state_dim) arrays
            Rollout states for each candidate.
        window_start : int
            Start of the window (unused, kept for interface symmetry).
        window_end : int
            End of the window — positions at this index are compared.

        Returns
        -------
        float in [0, 1] — multimodality proxy.
        """
        if len(candidate_states) <= 1:
            return 0.0
        idx = max(1, min(window_end, candidate_states[0].shape[0] - 1))
        pts = np.array(
            [np.asarray(s[idx, : self._position_dim], dtype=np.float32).ravel()
             for s in candidate_states],
            dtype=np.float32,
        )
        if pts.shape[0] <= 1:
            return 0.0
        spread = float(np.mean(np.std(pts, axis=0)))
        return float(np.clip(spread / self._multi_scale, 0.0, 1.0))
