"""
Dual-control diffusion scheduler.

Adaptively adjusts diffusion parameters based on diversity (ESS) feedback
using a dual variable λ^diff that tracks target ESS.
"""

from typing import Optional, Dict, Any, Callable
import numpy as np

from enerdynamics.core.constraints.schedulers.DiffusionScheduler.base import DiffusionScheduler
from enerdynamics.core.constraints.core.types import ScheduleState
from enerdynamics.core.constraints.core.registry import get_registry


class DualControlDiffusionScheduler(DiffusionScheduler):
    """
    Dual-control diffusion scheduler.
    
    Uses dual variable λ^diff to adaptively control diffusion parameters
    based on diversity (ESS) feedback. The dual variable is updated using
    Robbins-Monro style stochastic approximation to track a target ESS schedule.
    
    Key features:
    - Adaptive sample size M_k based on λ^diff
    - Adaptive temperature T_k based on λ^diff
    - Optional beta scaling s_k based on λ^diff
    - Progress-based scheduling
    
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
        lambda_diff_min: float = -2.0,
        lambda_diff_max: float = 2.0,
        lambda_diff_init: float = 0.0,
        eta_diff_base: float = 0.01,  # Much smaller than eta_con
        eta_diff_decay: float = 0.99,
        
        # Target ESS schedule
        ess_star_min: float = 0.3,
        ess_star_max: float = 0.8,
        p_ess: float = 1.0,
        
        # Diffusion mapping parameters
        M_min: int = 32,
        M_max: int = 512,
        c_M: float = 0.5,
        T_min: float = 0.1,
        T_max: float = 2.0,
        a_T: float = 1.0,
        b_T: float = 0.0,
        
        # Optional beta scaling
        enable_beta_scaling: bool = False,
        c_beta: float = 0.1,
        s_min: float = 0.5,
        s_max: float = 2.0,
        
        backend: str = "numpy",
        _skip_backend_lookup: bool = False,
        **kwargs
    ):
        """
        Initialize dual-control diffusion scheduler.
        
        Args:
            diffusion_schedule: Pre-computed DiffusionNoiseSchedule
            betas: Beta schedule array (alternative to diffusion_schedule)
            snr_min: Minimum SNR for normalization (auto-computed if None)
            snr_max: Maximum SNR for normalization (auto-computed if None)
            lambda_diff_min: Minimum dual variable value
            lambda_diff_max: Maximum dual variable value
            lambda_diff_init: Initial dual variable value
            eta_diff_base: Base step size for dual update (much smaller than eta_con)
            eta_diff_decay: Step size decay factor (per step)
            ess_star_min: Initial target ESS
            ess_star_max: Final target ESS
            p_ess: Power for cosine annealing of ESS*(t)
            M_min: Minimum sample size
            M_max: Maximum sample size
            c_M: Scale parameter for M_k exponential mapping
            T_min: Minimum temperature
            T_max: Maximum temperature
            a_T: Scale parameter for T_k sigmoid mapping
            b_T: Offset parameter for T_k sigmoid mapping
            enable_beta_scaling: Whether to enable beta scaling
            c_beta: Scale parameter for beta scaling exponential
            s_min: Minimum beta scaling factor
            s_max: Maximum beta scaling factor
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
        self.lambda_diff = float(lambda_diff_init)
        self.lambda_diff_min = float(lambda_diff_min)
        self.lambda_diff_max = float(lambda_diff_max)
        self.eta_diff_base = float(eta_diff_base)
        self.eta_diff_decay = float(eta_diff_decay)
        self.step_count = 0
        
        # Target ESS schedule parameters
        self.ess_star_min = float(ess_star_min)
        self.ess_star_max = float(ess_star_max)
        self.p_ess = float(p_ess)
        
        # Diffusion mapping parameters
        self.M_min = int(M_min)
        self.M_max = int(M_max)
        self.c_M = float(c_M)
        self.T_min = float(T_min)
        self.T_max = float(T_max)
        self.a_T = float(a_T)
        self.b_T = float(b_T)
        
        # Beta scaling parameters
        self.enable_beta_scaling = bool(enable_beta_scaling)
        self.c_beta = float(c_beta)
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        
        self.backend = backend
        self.kwargs = kwargs
        
        # Pre-compute constants for performance
        self._T_range = self.T_max - self.T_min
        self._ess_range = self.ess_star_max - self.ess_star_min
        self._s_range = self.s_max - self.s_min
        
        # Get backend implementation
        if not _skip_backend_lookup:
            registry = get_registry()
            impl_class = registry.get("scheduler", "dual_control_diffusion", backend)
            if impl_class is not None and impl_class != DualControlDiffusionScheduler:
                self._backend_impl = impl_class(
                    diffusion_schedule=self.diffusion_schedule,
                    snr_min=snr_min, snr_max=snr_max,
                    lambda_diff_min=lambda_diff_min, lambda_diff_max=lambda_diff_max,
                    lambda_diff_init=lambda_diff_init,
                    eta_diff_base=eta_diff_base, eta_diff_decay=eta_diff_decay,
                    ess_star_min=ess_star_min, ess_star_max=ess_star_max, p_ess=p_ess,
                    M_min=M_min, M_max=M_max, c_M=c_M,
                    T_min=T_min, T_max=T_max, a_T=a_T, b_T=b_T,
                    enable_beta_scaling=enable_beta_scaling,
                    c_beta=c_beta, s_min=s_min, s_max=s_max,
                    backend=backend, _skip_backend_lookup=True, **kwargs
                )
            else:
                self._backend_impl = None
        else:
            self._backend_impl = None
    
    def diffusion_params(self, state: ScheduleState) -> Dict[str, Any]:
        """
        Generate adaptive diffusion parameters.
        
        Computes diffusion parameters based on:
        - Dual variable λ^diff (from ESS feedback)
        - Optional: diffusion progress t_k
        
        Args:
            state: Current schedule state
            
        Returns:
            Dictionary with adaptive diffusion parameters
        """
        if self._backend_impl is not None:
            return self._backend_impl.diffusion_params(state)
        
        # Fallback implementation (should use backend)
        return self._compute_diffusion_params(state)
    
    def _compute_diffusion_params(self, state: ScheduleState) -> Dict[str, Any]:
        """
        Compute diffusion parameters (to be implemented in backend).
        
        This is a placeholder - actual implementation should be in backend.
        """
        raise NotImplementedError("Should use backend implementation")
    
    def update(self, state: ScheduleState, feedback: Dict[str, Any]) -> None:
        """
        Update dual variable based on ESS feedback.
        
        Implements Robbins-Monro style stochastic approximation:
        λ_{k+1} = clip(λ_k + η_k * (ESS*(t_k) - ESŜ_k), [λ_min, λ_max])
        
        Args:
            state: Current schedule state
            feedback: Dictionary with "ess" or "ess_hat" key
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
        self.lambda_diff = 0.0
        self.step_count = 0
        if self._backend_impl is not None and hasattr(self._backend_impl, 'reset'):
            self._backend_impl.reset()
    
    def get_scaled_betas(self) -> Optional[np.ndarray]:
        """
        Get scaled beta schedule if scaling is enabled.
        
        Returns:
            Scaled betas array (β'_k = s_k * β_k) or None if scaling disabled
        """
        if not self.enable_beta_scaling:
            return None
        
        if self._backend_impl is not None and hasattr(self._backend_impl, 'get_scaled_betas'):
            return self._backend_impl.get_scaled_betas()
        
        # Compute scaling factor
        s_k = np.clip(
            np.exp(self.c_beta * self.lambda_diff),
            self.s_min,
            self.s_max
        )
        return self.diffusion_schedule.betas * s_k

