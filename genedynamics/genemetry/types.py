"""
Pure data structures for constraint geometry.

All array fields use ``Any`` so the types stay backend-agnostic.
No backend imports (numpy, jax, torch) appear at this level.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class GeometryBundle:
    """Local constraint geometry at a reference trajectory.

    Encapsulates the linearized constraint structure needed for
    projection and retraction operations.

    Fields
    ------
    A_rows : (K, D)
        Normalized active constraint basis vectors.
    topu : (K,)
        Row norms of the top-k constraint vectors (before normalization).
    active_count : scalar
        Number of active constraints (float for JAX traceability).
    is_valid : scalar bool
        Whether there is at least one active constraint.
    metric_sys : (K, K)
        SPD system matrix for metric projection:
        ``I + A A^T + eps * I``.
    tan_sys : (K, K)
        SPD system matrix for tangent projection:
        ``A A^T + eps * I``.
    raw_geometry : (H, D) or None
        The (possibly windowed / modulated) constraint proxy vectors
        that produced this bundle.
    meta : dict
        Auxiliary data (``idx_top``, ``active_mask_t``, ``score_base``, ...).
    """

    A_rows: Any
    topu: Any
    active_count: Any
    is_valid: Any
    metric_sys: Any
    tan_sys: Any
    raw_geometry: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GateDecision:
    """Decision from a gating policy.

    Fields
    ------
    gamma : scalar float in [0, 1]
        Gating coefficient.  0 = skip geometry, 1 = full geometry.
    meta : dict
        Auxiliary diagnostics (e.g. ``pi_multi``, ``spread``).
    """

    gamma: Any
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetractionResult:
    """Result of a retraction operation.

    Fields
    ------
    trajectory : same shape as input
        The retracted (feasible-projected) trajectory.
    meta : dict
        Auxiliary data (e.g. number of QP iterations).
    """

    trajectory: Any
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeResult:
    """Result of a probe mini-batch pipeline.

    Fields
    ------
    fixed_trajectories : (N, H, D)
        Sample set with probe-retracted entries scattered back in.
    residual_geometry : (H, D)
        Mean residual (retracted - original) averaged over probed
        trajectories.  Used as probe geometry correction.
    meta : dict
        Auxiliary diagnostics (e.g. ``n_probed``, ``bmask``).
    """

    fixed_trajectories: Any
    residual_geometry: Any
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RefinementResult:
    """Result of a post-hoc refinement pipeline.

    Fields
    ------
    candidate_actions : list of (H, U) arrays
        Refined action candidates.
    candidate_states : list of (H+1, state_dim) arrays
        Rollout states for refined candidates.
    candidate_costs : (C,) array
        Costs after refinement.
    best_idx : int
        Index of the best candidate.
    window_gamma_hist : (W,) array
        Per-window gating decisions.
    window_cvar_hist : (W,) array
        Per-window CVaR values.
    window_delta_hist : (W,) array
        Per-window delta thresholds.
    refined_indices : array
        Indices of candidates that were modified.
    meta : dict
        Additional diagnostics.
    """

    candidate_actions: Any
    candidate_states: Any
    candidate_costs: Any
    best_idx: int
    window_gamma_hist: Any
    window_cvar_hist: Any
    window_delta_hist: Any
    refined_indices: Any
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result of a constrained optimization step.

    Fields
    ------
    actions : (H, U)
        Updated actions after the constrained step.
    meta : dict
        Auxiliary data (e.g. ``n_active``, ``accepted``).
    """

    actions: Any
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OverlayParams:
    """Per-step overlay parameters produced by the schedule overlay.

    Fields
    ------
    kappa : scalar
        Constraint penalty weight for AGP update.
    delta : scalar
        CVaR violation threshold for retraction triggering.
    sigma : scalar
        Diffusion noise scale.
    theta : scalar
        Gating threshold for multimodality proxy.
    eta : scalar
        AGP step size (eta_base * eta_scale).
    hardness : scalar
        Constraint difficulty in [0, 1].
    meta : dict
        Auxiliary data.
    """

    kappa: Any
    delta: Any
    sigma: Any
    theta: Any
    eta: Any
    hardness: Any = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)
