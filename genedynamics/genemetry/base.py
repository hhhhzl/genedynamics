"""
Abstract interfaces for constraint geometry components.

These ABCs define the contract between solvers and the geometry layer.
Concrete implementations are registered per-backend via the genemetry
registry and dispatched through thin orchestrator classes.

Design mapping to rieoptax manifold concepts
---------------------------------------------
=========================  ===========================
rieoptax                   genemetry
=========================  ===========================
Manifold.exp / .retr       RetractionOperator.retract
Manifold.log               (inverse retraction, future)
Manifold.egrad_to_rgrad    ConstraintManifold.project
Manifold.inp               (implicit via metric_sys)
Manifold.dist              (implicit via geometry)
NFoldManifold              ProductManifold (future)
optimizer transform        solver scan body
=========================  ===========================
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from genedynamics.genemetry.types import (
    GeometryBundle,
    GateDecision,
    RetractionResult,
    ProbeResult,
    RefinementResult,
    StepResult,
    OverlayParams,
)


# ======================================================================
# Constraint manifold
# ======================================================================

class ConstraintManifold(ABC):
    """Local constraint manifold for trajectory optimization.

    Computes the linearized constraint geometry (active basis, metric
    tensor, tangent-space structure) and provides projection in the
    constraint-adapted metric.
    """

    @abstractmethod
    def geometry(
        self,
        constraint_vectors: Any,
        topk_active: int,
        eps_stab: float,
    ) -> GeometryBundle:
        """Compute local geometry from constraint proxy vectors.

        Parameters
        ----------
        constraint_vectors : (H, D)
            Per-timestep constraint proxy vectors (e.g. windowed SDF
            normals, already modulated by task modulator if applicable).
        topk_active : int
            Number of top-k active constraint directions to retain.
        eps_stab : float
            Numerical stabilization constant.

        Returns
        -------
        GeometryBundle
        """

    @abstractmethod
    def project(
        self,
        vectors: Any,
        bundle: GeometryBundle,
        mode: str = "metric",
    ) -> Any:
        """Project vectors through the constraint complement.

        ``mode="metric"``
            Uses ``bundle.metric_sys`` (I + A A^T + eps I).
            Suitable for score / gradient projection.

        ``mode="tangent"``
            Uses ``bundle.tan_sys`` (A A^T + eps I).
            Suitable for noise projection.

        Parameters
        ----------
        vectors : (B, D)
            Batch of vectors to project.
        bundle : GeometryBundle
            Geometry computed by :meth:`geometry`.
        mode : ``"metric"`` | ``"tangent"``

        Returns
        -------
        Projected vectors, same shape as *vectors*.
        """


# ======================================================================
# Retraction operator
# ======================================================================

class RetractionOperator(ABC):
    """Maps a trajectory back to the feasible set.

    Analogous to ``retr()`` in rieoptax but operates on full
    trajectories rather than single tangent vectors.
    """

    @abstractmethod
    def retract(
        self,
        state: Any,
        trajectory: Any,
        params: Optional[Any] = None,
    ) -> RetractionResult:
        """Retract *trajectory* starting from *state*.

        Parameters
        ----------
        state : initial state (e.g. x0)
        trajectory : (H, act_dim) proposed action sequence
        params : backend- / method-specific parameters
            For CFS: ``{"sched_state": ..., "sched_params": ...}``
            For stepping: unused (may be ``None``).

        Returns
        -------
        RetractionResult
        """


# ======================================================================
# Gate policy
# ======================================================================

class GatePolicy(ABC):
    """Decides when to activate geometric operations.

    Returns a scalar gating coefficient gamma in [0, 1] that modulates
    the strength of geometry-driven corrections.
    """

    @abstractmethod
    def evaluate(
        self,
        samples: Any,
        reference: Any,
        window_mask: Any,
        sigma_scale: Any,
        theta: Any,
    ) -> GateDecision:
        """Evaluate the gating condition.

        Parameters
        ----------
        samples : (N, H, D)
            Current sample set.
        reference : (H, D)
            Current reference trajectory (mean).
        window_mask : (H,)
            Binary window mask over the horizon.
        sigma_scale : scalar
            Current diffusion noise level.
        theta : scalar
            Gating threshold.

        Returns
        -------
        GateDecision
        """


# ======================================================================
# Task modulator
# ======================================================================

class TaskModulator(ABC):
    """Adapts raw constraint geometry based on task context.

    Operates on the raw constraint proxy vectors *before* they are
    passed to :meth:`ConstraintManifold.geometry`.
    """

    @abstractmethod
    def modulate(
        self,
        raw_geometry: Any,
        probe_geometry: Optional[Any] = None,
        window_mask: Optional[Any] = None,
        alpha: float = 0.0,
    ) -> Any:
        """Return modulated constraint proxy vectors.

        Parameters
        ----------
        raw_geometry : (H, D)
            Base (windowed) constraint proxy.
        probe_geometry : (H, D) or None
            Probe-derived geometry correction.
        window_mask : (H,) or None
            Window mask (used to mask probe contribution).
        alpha : float
            Blending coefficient for the probe geometry.

        Returns
        -------
        Modulated constraint vectors, same shape as *raw_geometry*.
        """


# ======================================================================
# Window policy
# ======================================================================

class WindowPolicy(ABC):
    """Defines a windowing scheme over the planning horizon.

    Windows focus geometric operations on a subset of the horizon,
    improving both computational efficiency and constraint locality.
    """

    @abstractmethod
    def mask(
        self,
        step_k: Any,
        horizon: int,
    ) -> Any:
        """Compute a binary window mask for diffusion step *step_k*.

        Parameters
        ----------
        step_k : scalar int
            Current diffusion step index.
        horizon : int
            Planning horizon length.

        Returns
        -------
        (H,) float mask — 1.0 inside window, 0.0 outside.
        """

    @abstractmethod
    def slices(self, horizon: int) -> List[Tuple[int, int]]:
        """Return a list of ``(start, end)`` window slices for refinement.

        Parameters
        ----------
        horizon : int
            Planning horizon length.

        Returns
        -------
        List of (start, end) tuples covering the full horizon.
        """


# ======================================================================
# Probe sampler
# ======================================================================

class ProbeSampler(ABC):
    """Mini-batch probe pipeline: sample → retract → extract residual.

    Probes a subset of trajectories through the retraction operator
    to extract a residual geometry correction (probe geometry).
    """

    @abstractmethod
    def sample_and_retract(
        self,
        trajectories: Any,
        costs: Any,
        retraction_op: "RetractionOperator",
        state: Any,
        retract_params: Any,
        rng_keys: Any,
    ) -> ProbeResult:
        """Run probe mini-batch and extract residual geometry.

        Parameters
        ----------
        trajectories : (N, H, D)
            Full set of sampled trajectories.
        costs : (N,)
            Per-trajectory cost or violation scores.
        retraction_op : RetractionOperator
            Retraction to apply to probed trajectories.
        state : initial state (x0)
        retract_params : dict
            Parameters for the retraction operator.
        rng_keys : backend-specific RNG state
            Keys for tail/random subset selection.

        Returns
        -------
        ProbeResult
        """


# ======================================================================
# Constrained step
# ======================================================================

class ConstrainedStep(ABC):
    """Single constrained optimization step in action space.

    Given constraint violations and gradients, computes a feasibility-
    improving update using the constraint geometry.  Analogous to a
    single Riemannian gradient step in rieoptax.
    """

    @abstractmethod
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
        """Compute one constrained update step.

        Parameters
        ----------
        actions : (H, U)
            Current action sequence.
        violations : (H,)
            Per-timestep constraint violations (g_plus = max(0, margin - sdf)).
        gradients : (H, dim)
            Per-timestep constraint gradients.
        window : (start, end)
            Active window within the horizon.
        sigma : float
            Noise scale for tangent noise injection.
        kappa : float
            Constraint penalty weight.
        eta : float
            Step size.

        Returns
        -------
        StepResult
        """


# ======================================================================
# Refinement pipeline
# ======================================================================

class RefinementPipeline(ABC):
    """Post-hoc window-by-window trajectory refinement.

    Iterates over sliding windows, evaluating constraint violations
    and applying constrained steps + local retraction to improve
    feasibility of candidate trajectories.
    """

    @abstractmethod
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
        """Refine candidates via window-local constrained steps.

        Parameters
        ----------
        candidate_actions : list of (H, U) arrays
            Action candidates to refine.
        candidate_states : list of (H+1, state_dim) arrays
            Corresponding rollout states.
        violation_fn : callable
            ``(states, clearance) -> (g_plus, grad)``
            where g_plus is (H,) violations and grad is (H, dim).
        cost_fn : callable
            ``(actions) -> float`` cost of an action sequence.
        schedule_params : dict
            Keys: ``sigma_hist``, ``delta_hist``, ``theta_hist``,
            ``eta_hist``, ``kappa_hist`` — arrays of length K.
        clearance : float
            Minimum SDF clearance for violation computation.

        Returns
        -------
        RefinementResult
        """


# ======================================================================
# Schedule overlay
# ======================================================================

class ScheduleOverlayBase(ABC):
    """Maps constraint state (rho, margin) to per-step solver parameters.

    The overlay chain:
    1. **Hardness**: ``rho -> hardness in [0, 1]`` via log-scale normalization.
    2. **Constraint overlay**: ``(margin, rho) -> (kappa, delta)``.
    3. **Diffusion overlay**: ``(hardness, eta_base) -> (sigma, theta, eta)``.

    Supports both element-wise (per-step in JAX scan) and batch (history
    reconstruction in NumPy) usage.
    """

    @abstractmethod
    def hardness(self, rho: Any) -> Any:
        """Map augmented Lagrangian penalty to hardness in [0, 1].

        Parameters
        ----------
        rho : scalar or array
            Augmented Lagrangian penalty.

        Returns
        -------
        Hardness in [0, 1], same shape as *rho*.
        """

    @abstractmethod
    def constraint_overlay(self, margin: Any, rho: Any) -> Tuple[Any, Any]:
        """Compute constraint penalty weight and violation threshold.

        Parameters
        ----------
        margin : scalar or array — SDF clearance margin.
        rho : scalar or array — augmented Lagrangian penalty.

        Returns
        -------
        (kappa, delta) — penalty weight and CVaR threshold.
        """

    @abstractmethod
    def diffusion_overlay(
        self, hardness: Any, eta_base: float
    ) -> Tuple[Any, Any, Any]:
        """Compute diffusion parameters from hardness.

        Parameters
        ----------
        hardness : scalar or array in [0, 1].
        eta_base : float — base AGP step size.

        Returns
        -------
        (sigma, theta, eta) — noise scale, gate threshold, step size.
        """

    def compute(
        self, margin: Any, rho: Any, eta_base: float
    ) -> OverlayParams:
        """Full overlay chain: (margin, rho) -> OverlayParams.

        Convenience method that composes hardness, constraint_overlay,
        and diffusion_overlay in sequence.
        """
        h = self.hardness(rho)
        kappa, delta = self.constraint_overlay(margin, rho)
        sigma, theta, eta = self.diffusion_overlay(h, eta_base)
        return OverlayParams(
            kappa=kappa, delta=delta,
            sigma=sigma, theta=theta, eta=eta,
            hardness=h,
        )

    def compute_series(
        self, margin_hist: Any, rho_hist: Any, eta_base: float
    ) -> OverlayParams:
        """Batch overlay over full histories (for post-hoc diagnostics).

        Same interface as :meth:`compute` but operates on arrays.
        """
        return self.compute(margin_hist, rho_hist, eta_base)
