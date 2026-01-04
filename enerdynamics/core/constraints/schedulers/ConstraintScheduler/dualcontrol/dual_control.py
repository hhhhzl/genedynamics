"""
Dual-control constraint scheduler.

Adaptively adjusts constraint parameters based on feasibility feedback
using a dual variable λ^con that tracks target feasibility.
"""

from typing import Optional, Dict, Any, Callable, List
import numpy as np

from enerdynamics.core.constraints.schedulers.ConstraintScheduler.base import ConstraintScheduler
from enerdynamics.core.constraints.core.types import ScheduleState
from enerdynamics.core.constraints.core.registry import get_registry


class DualControlConstraintScheduler(ConstraintScheduler):
    """
    Dual-control constraint scheduler.
    
    Uses dual variable λ^con (feasibility tracking variable) to adaptively control 
    constraint hardness based on target feasibility tracking. The dual variable is 
    updated using Robbins-Monro style stochastic approximation to track a target
    feasibility schedule q*(s).
    
    Note: λ^con is a signed dual variable (can be negative) that tracks feasibility
    gap (q* - q_hat). It is NOT a standard Lagrange multiplier (which would be ≥ 0).
    When q_hat > q*, λ^con decreases (can go negative); when q_hat < q*, λ^con increases.
    This allows the scheduler to adapt both when constraints are too loose (negative λ)
    and when they are too tight (positive λ).
    
    Key features:
    - Unified tightening progress s_k = 1 - t_k drives all scheduling
    - Continuous cosine annealing interpolation for all parameters
    - Adaptive slack penalty ρ(s_k) based on s_k
    - Adaptive QP gate probability p(s_k) based on s_k (approaches 1.0 at later steps)
    - Progress-based scheduling for topK, eps, I_QP using s_k
    - Terminal hard region: last 1-3 steps force gate=ON
    
    Performance: O(1) per parameter computation, O(1) dual update.
    """
    
    def __init__(
        self,
        # Diffusion schedule (for progress calculation)
        diffusion_schedule: Optional[Any] = None,
        betas: Optional[np.ndarray] = None,
        snr_min: Optional[float] = None,
        snr_max: Optional[float] = None,
        
        # Dual variable parameters
        lambda_con_min: float = -2.0,
        lambda_con_max: float = 2.0,
        lambda_con_init: float = 0.0,
        eta_con_base: float = 0.1,
        eta_con_decay: float = 0.99,
        
        # Target feasibility schedule
        q_star_min: float = 0.3,
        q_star_max: float = 0.95,
        p_q: float = 1.0,
        
        # Constraint mapping parameters
        rho_min: float = 0.1,
        rho_max: float = 100.0,
        a_rho: float = 1.0,
        b_rho: float = 0.0,
        a_g: float = 1.0,
        b_g: float = 0.0,
        
        # Constraint scheduling parameters
        K_min: int = 5,
        K_max: int = 50,
        eps_min: float = 1e-4,
        eps_max: float = 1e-2,
        p_eps: float = 1.0,
        I_min: int = 1,
        I_max: int = 10,
        
        # Terminal hard region
        t_hard: float = 0.8,
        
        backend: str = "numpy",
        _skip_backend_lookup: bool = False,
        **kwargs
    ):
        """
        Initialize dual-control constraint scheduler.
        
        Args:
            diffusion_schedule: Pre-computed DiffusionNoiseSchedule
            betas: Beta schedule array (alternative to diffusion_schedule)
            snr_min: Minimum SNR for normalization (auto-computed if None)
            snr_max: Maximum SNR for normalization (auto-computed if None)
            lambda_con_min: Minimum dual variable value
            lambda_con_max: Maximum dual variable value
            lambda_con_init: Initial dual variable value
            eta_con_base: Base step size for dual update
            eta_con_decay: Step size decay factor (per step)
            q_star_min: Initial target feasibility rate
            q_star_max: Final target feasibility rate
            p_q: Power for cosine annealing of q*(t)
            rho_min: Minimum slack penalty
            rho_max: Maximum slack penalty
            a_rho: Scale parameter for rho sigmoid mapping
            b_rho: Offset parameter for rho sigmoid mapping
            a_g: Scale parameter for gate probability sigmoid
            b_g: Offset parameter for gate probability sigmoid
            K_min: Minimum active constraint count
            K_max: Maximum active constraint count
            eps_min: Minimum solver tolerance (fine)
            eps_max: Maximum solver tolerance (coarse)
            p_eps: Power for tolerance annealing
            I_min: Minimum QP iterations
            I_max: Maximum QP iterations
            t_hard: Terminal hard region threshold (t_k >= t_hard)
            backend: Backend to use ("numpy", "jax")
            _skip_backend_lookup: Internal flag (prevents recursion)
            **kwargs: Additional parameters
        """
        # Setup diffusion schedule
        if diffusion_schedule is not None:
            self.diffusion_schedule = diffusion_schedule
        elif betas is not None:
            from enerdynamics.core.constraints.schedulers.utils import DiffusionNoiseSchedule
            self.diffusion_schedule = DiffusionNoiseSchedule.from_betas(betas)
        else:
            raise ValueError("Must provide either diffusion_schedule or betas")
        
        # Compute SNR bounds (cached for performance)
        if snr_min is None or snr_max is None:
            snr_min_computed, snr_max_computed = self.diffusion_schedule.compute_snr_bounds()
            if snr_min is None:
                snr_min = snr_min_computed
            if snr_max is None:
                snr_max = snr_max_computed
        
        self.snr_min = float(snr_min)
        self.snr_max = float(snr_max)
        
        # Store dual variable parameters
        self.lambda_con = float(lambda_con_init)
        self.lambda_con_min = float(lambda_con_min)
        self.lambda_con_max = float(lambda_con_max)
        self.eta_con_base = float(eta_con_base)
        self.eta_con_decay = float(eta_con_decay)
        self.step_count = 0
        
        # Store current feasibility rate for modulation (Solution 2)
        # Initialize to q_star_min to match q_star's starting point
        self.q_hat = float(q_star_min)  # Initialize to q_star_min (e.g., 0.3) to match q_star
        self.q_tilde = float(q_star_min)  # Synthesized feasibility (q̃_k), start at q_star_min
        self.V_k = 0.0  # Current violation V_k
        self.V_k_prev = 0.0  # Previous violation V_{k-1}
        self.q_hat_prev = float(q_star_min)  # Previous q_hat, start at q_star_min
        
        # Target schedule parameters
        self.q_star_min = float(q_star_min)
        self.q_star_max = float(q_star_max)
        self.p_q = float(p_q)
        
        # Constraint mapping parameters
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.a_rho = float(a_rho)
        self.b_rho = float(b_rho)
        self.a_g = float(a_g)
        self.b_g = float(b_g)
        
        # Constraint scheduling parameters
        self.K_min = int(K_min)
        self.K_max = int(K_max)
        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)
        self.p_eps = float(p_eps)
        self.I_min = int(I_min)
        self.I_max = int(I_max)
        self.t_hard = float(t_hard)
        
        self.backend = backend
        self.kwargs = kwargs
        
        # History tracking for visualization
        self.param_history = []
        self.lambda_con_history = []
        
        # Pre-compute constants for performance
        self._rho_range = self.rho_max - self.rho_min
        self._K_range = self.K_max - self.K_min
        self._eps_range = self.eps_max - self.eps_min
        self._I_range = self.I_max - self.I_min
        self._q_range = self.q_star_max - self.q_star_min
        
        # Get backend implementation
        if not _skip_backend_lookup:
            registry = get_registry()
            impl_class = registry.get("scheduler", "dual_control_constraint", backend)
            if impl_class is not None and impl_class != DualControlConstraintScheduler:
                self._backend_impl = impl_class(
                    diffusion_schedule=self.diffusion_schedule,
                    snr_min=snr_min, snr_max=snr_max,
                    lambda_con_min=lambda_con_min, lambda_con_max=lambda_con_max,
                    lambda_con_init=lambda_con_init,
                    eta_con_base=eta_con_base, eta_con_decay=eta_con_decay,
                    q_star_min=q_star_min, q_star_max=q_star_max, p_q=p_q,
                    rho_min=rho_min, rho_max=rho_max,
                    a_rho=a_rho, b_rho=b_rho, a_g=a_g, b_g=b_g,
                    K_min=K_min, K_max=K_max,
                    eps_min=eps_min, eps_max=eps_max, p_eps=p_eps,
                    I_min=I_min, I_max=I_max, t_hard=t_hard,
                    backend=backend, _skip_backend_lookup=True, **kwargs
                )
            else:
                self._backend_impl = None
        else:
            self._backend_impl = None
    
    def constraint_params(self, state: ScheduleState) -> Dict[str, Any]:
        """
        Generate adaptive constraint parameters.
        
        Computes constraint parameters based on:
        - Diffusion progress t_k (from SNR)
        - Dual variable λ^con (from feasibility feedback)
        - Terminal hard region override
        
        Args:
            state: Current schedule state
            
        Returns:
            Dictionary with adaptive constraint parameters
        """
        if self._backend_impl is not None:
            params = self._backend_impl.constraint_params(state)
        else:
        # Fallback implementation (should use backend)
            params = self._compute_constraint_params(state)
        
        # Record history
        self._record_param_history(state, params)
        
        return params
    
    def _record_param_history(self, state: ScheduleState, params: Dict[str, Any]) -> None:
        """
        Record parameter history for visualization.
        
        Args:
            state: Current schedule state
            params: Parameter dictionary
        """
        extra = params.get("_extra", {})
        self.param_history.append({
            "step": state.k,
            "total_steps": state.K,
            "rho": params.get("rho", 0.0),
            "topK": params.get("topK"),
            "eps": params.get("eps", 0.0),
            "I_QP": params.get("I_QP", 0),
            "qp_gate": params.get("qp_gate", False),
            "qp_prob": params.get("qp_prob", 0.0),
            "lambda_con": extra.get("lambda_con", self.lambda_con),
            "t_k": extra.get("t_k", 0.0),
            "q_star": extra.get("q_star", 0.0),
            "q_hat": extra.get("q_hat", self.q_hat),  # Add q_hat to history
            "q_tilde": extra.get("q_tilde", self.q_tilde),  # Add q_tilde to history
            "V_k": extra.get("V_k", self.V_k),  # Add V_k to history
            "is_terminal_hard": extra.get("is_terminal_hard", False),
            "_extra": extra,  # Also store full _extra dict for backward compatibility
        })
        self.lambda_con_history.append(self.lambda_con)
    
    def _compute_constraint_params(self, state: ScheduleState) -> Dict[str, Any]:
        """
        Compute constraint parameters (to be implemented in backend).
        
        This is a placeholder - actual implementation should be in backend.
        """
        raise NotImplementedError("Should use backend implementation")
    
    def update(self, state: ScheduleState, feedback: Dict[str, Any]) -> None:
        """
        Update dual variable based on feasibility feedback.
        
        Implements Robbins-Monro style stochastic approximation:
        λ_{k+1} = clip(λ_k + η_k * (q*(t_k) - q̂_k), [λ_min, λ_max])
        
        Args:
            state: Current schedule state
            feedback: Dictionary with "feasible_rate" or "q_hat" key
        """
        if self._backend_impl is not None:
            self._backend_impl.update(state, feedback)
        else:
            self._update_dual_variable(state, feedback)
    
    def _update_dual_variable(self, state: ScheduleState, feedback: Dict[str, Any]) -> None:
        """
        Update dual variable (to be implemented in backend).
        
        This is a placeholder - actual implementation should be in backend.
        """
        pass
    
    def reset(self) -> None:
        """Reset scheduler state (for new optimization run)."""
        self.lambda_con = float(self.lambda_con_init)
        self.step_count = 0
        # Reset to q_star_min to match q_star's starting point
        self.q_hat = float(self.q_star_min)  # Reset to q_star_min (e.g., 0.3) to match q_star
        self.q_tilde = float(self.q_star_min)  # Reset synthesized feasibility to q_star_min
        self.V_k = 0.0  # Reset violation
        self.V_k_prev = 0.0  # Reset previous violation
        self.q_hat_prev = float(self.q_star_min)  # Reset previous q_hat to q_star_min
        self.param_history = []
        self.lambda_con_history = []
        if self._backend_impl is not None and hasattr(self._backend_impl, 'reset'):
            self._backend_impl.reset()
    
    def get_param_history(self) -> List[Dict[str, Any]]:
        """
        Get parameter history for visualization.
        
        Returns:
            List of parameter dictionaries, one per diffusion step
        """
        return self.param_history.copy()
    
    def get_lambda_con_history(self) -> List[float]:
        """
        Get lambda_con history for visualization.
        
        Returns:
            List of lambda_con values, one per diffusion step
        """
        return self.lambda_con_history.copy()

