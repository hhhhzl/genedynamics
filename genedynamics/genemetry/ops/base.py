"""
Abstract interface for low-level geometry operations.

Backend implementations (JAX, NumPy, ...) inherit from :class:`GeometryOps`
and register themselves via :func:`register_genemetry`.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple


class GeometryOps(ABC):
    """Low-level geometry primitives required by constraint manifolds.

    Each method is stateless: it takes array inputs and returns array
    outputs.  Implementations are backend-specific and registered via
    ``@register_genemetry("ops", "geometry", "<backend>")``.
    """

    @abstractmethod
    def build_active_rows(
        self,
        a_time: Any,
        topk_active: int,
        eps_stab: float,
    ) -> Tuple[Any, Any, Any, Any, Any]:
        """Build low-rank active geometry basis from constraint proxy vectors.

        Parameters
        ----------
        a_time : (H, D)
            Per-timestep constraint proxy vectors.
        topk_active : int
            Number of top-k directions to retain.
        eps_stab : float
            Numerical stabilization constant.

        Returns
        -------
        A_rows : (K, D)
            Normalized active constraint rows.
        topu : (K,)
            Top-k row norms (pre-normalization).
        idx_top : (K,)
            Indices of the top-k time steps.
        active_mask_t : (H,)
            Binary mask indicating active time steps.
        score_base : (H, D)
            Masked geometry proxy (``a_time * active_mask``).
        """

    @abstractmethod
    def project_complement_batch(
        self,
        vectors: Any,
        A_rows: Any,
        system: Any,
    ) -> Any:
        """Project a batch of vectors through I - A^T system^{-1} A.

        Uses a single Cholesky factorization of *system* for the whole
        batch.

        Parameters
        ----------
        vectors : (B, D)
        A_rows  : (K, D)
        system  : (K, K), SPD

        Returns
        -------
        Projected vectors, shape (B, D).
        """
