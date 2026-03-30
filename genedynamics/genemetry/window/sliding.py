"""
Sliding window policy for horizon-local geometry operations.

Backend-agnostic: the mask can be computed with any array library
that supports basic indexing and comparison operators.  The slices
method is pure Python.
"""

from typing import Any, List, Tuple

from genedynamics.genemetry.base import WindowPolicy


class SlidingWindow(WindowPolicy):
    """Fixed-size sliding window that advances with diffusion step index.

    Parameters
    ----------
    window_size : int
        Number of horizon steps inside each window.
    window_stride : int
        Step advance between consecutive windows (for refinement slices)
        and the stride used for the per-diffusion-step cyclic mask.
    """

    def __init__(self, window_size: int, window_stride: int) -> None:
        self._size = max(1, int(window_size))
        self._stride = max(1, int(window_stride))

    # ------------------------------------------------------------------
    # WindowPolicy interface
    # ------------------------------------------------------------------

    def mask(self, step_k: Any, horizon: int) -> Any:
        """Cyclic binary mask centred at ``step_k * stride % horizon``.

        This method is designed to work inside JAX traced code:
        *step_k* may be a traced integer, *horizon* is a static int.

        The mask is 1.0 for timesteps inside the window and 0.0 outside.
        The window wraps around the horizon boundary.
        """
        # NOTE: we import inside the method so that the module stays
        # importable without JAX installed.  When called from NumPy
        # contexts, callers should use `mask_numpy` instead.
        import jax.numpy as jnp

        idxs = jnp.arange(horizon, dtype=jnp.int32)
        w_start = (step_k * self._stride) % horizon
        size = min(self._size, horizon)
        return (((idxs - w_start) % horizon) < size).astype(jnp.float32)

    def mask_numpy(self, step_k: int, horizon: int) -> Any:
        """Pure-NumPy variant for refinement (non-traced) contexts."""
        import numpy as np

        idxs = np.arange(horizon, dtype=np.int32)
        w_start = (int(step_k) * self._stride) % horizon
        size = min(self._size, horizon)
        return (((idxs - w_start) % horizon) < size).astype(np.float32)

    def slices(self, horizon: int) -> List[Tuple[int, int]]:
        """Non-overlapping window slices covering the full horizon."""
        W = min(self._size, horizon)
        S = self._stride
        result: List[Tuple[int, int]] = []
        i = 0
        while i < horizon:
            j = min(horizon, i + W)
            result.append((i, j))
            if j == horizon:
                break
            i += S
        return result

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def window_size(self) -> int:
        return self._size

    @property
    def window_stride(self) -> int:
        return self._stride

    def __repr__(self) -> str:
        return f"SlidingWindow(size={self._size}, stride={self._stride})"
