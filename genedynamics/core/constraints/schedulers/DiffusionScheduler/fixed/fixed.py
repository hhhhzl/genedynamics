"""
Fixed diffusion scheduler.

All diffusion parameters are constant regardless of diffusion step.
Useful for fixed sample size and temperature experiments.
"""

from typing import Optional, Dict, Any  
import numpy as np  

from genedynamics.core.constraints.schedulers.DiffusionScheduler.base import DiffusionScheduler
from genedynamics.core.constraints.core.types import ScheduleState
from genedynamics.core.constraints.core.registry import get_registry


class FixedDiffusionScheduler(DiffusionScheduler):
    """
    Fixed diffusion scheduler.
    
    All diffusion parameters are constant regardless of diffusion step.
    This scheduler provides deterministic behavior and is useful for:
    - Fixed sample size experiments
    - Fixed temperature experiments
    - Baseline comparisons
    - Testing and debugging
    
    Performance: O(1) - constant time parameter generation.
    """
    
    #: Recognised reverse-transport family tags (see
    #: genedynamics.solvers.common.transport). "DDPM" is the byte-identical
    #: default; "DDIM"/"FM" select the eps-form backends.
    _VALID_TRANSPORT_FAMILIES = ("DDPM", "DDIM", "FM")

    def __init__(
        self,
        M_k: int = 64,
        T_k: float = 0.5,
        s_k: Optional[float] = None,  # Optional beta scaling factor
        beta0: Optional[float] = None,
        betaT: Optional[float] = None,
        Ndiffuse: Optional[int] = None,
        transport_family: str = "DDPM",
        backend: str = "numpy",
        _skip_backend_lookup: bool = False,
        **kwargs
    ):
        """
        Initialize fixed diffusion scheduler.

        Args:
            M_k: Fixed sample size (number of candidates to generate)
            T_k: Fixed temperature for importance weighting
            s_k: Optional fixed beta scaling factor (None = no scaling)
            transport_family: Reverse-transport family selector emitted in
                ``diffusion_params`` (one of "DDPM" | "DDIM" | "FM"). Defaults
                to "DDPM", the byte-identical inline reverse update. A non-DDPM
                value makes the MBD solver build the matching eps-form transport
                (DDIM/FM) for the reverse loop.
            backend: Backend to use ("numpy", "jax") - for future JAX support
            _skip_backend_lookup: Internal flag to prevent recursion
            **kwargs: Additional parameters stored in _extra
        """
        # Store parameters as attributes for fast access
        self.M_k = int(M_k)
        self.T_k = float(T_k)
        self.s_k = float(s_k) if s_k is not None else None
        self.beta0 = float(beta0) if beta0 is not None else None
        self.betaT = float(betaT) if betaT is not None else None
        self.Ndiffuse = int(Ndiffuse) if Ndiffuse is not None else None
        self.transport_family = self._normalize_transport_family(transport_family)
        self.extra = kwargs
        self.backend = backend

        # Pre-compute parameter dict for fast access (performance optimization)
        self._cached_params = {
            "M_k": self.M_k,
            "T_k": self.T_k,
            "transport_family": self.transport_family,
            **self.extra,
        }
        if self.s_k is not None:
            self._cached_params["s_k"] = self.s_k
        if self.beta0 is not None:
            self._cached_params["beta0"] = self.beta0
        if self.betaT is not None:
            self._cached_params["betaT"] = self.betaT
        if self.Ndiffuse is not None:
            self._cached_params["Ndiffuse"] = self.Ndiffuse

        # Get backend implementation from registry (for future JAX support)
        if not _skip_backend_lookup:
            registry = get_registry()
            impl_class = registry.get("scheduler", "fixed_diffusion", backend)
            if impl_class is not None and impl_class != FixedDiffusionScheduler:
                self._backend_impl = impl_class(
                    M_k=M_k, T_k=T_k, s_k=s_k,
                    beta0=beta0, betaT=betaT, Ndiffuse=Ndiffuse,
                    transport_family=self.transport_family,
                    backend=backend, _skip_backend_lookup=True, **kwargs
                )
            else:
                self._backend_impl = None
        else:
            self._backend_impl = None

    @classmethod
    def _normalize_transport_family(cls, transport_family: Any) -> str:
        """Validate and upper-case the transport family tag (default DDPM)."""
        if transport_family is None:
            return "DDPM"
        fam = str(transport_family).upper()
        if fam not in cls._VALID_TRANSPORT_FAMILIES:
            raise ValueError(
                f"Unknown transport_family {transport_family!r}; expected one of "
                f"{cls._VALID_TRANSPORT_FAMILIES}."
            )
        return fam
    
    def diffusion_params(self, state: ScheduleState) -> Dict[str, Any]:
        """
        Generate fixed diffusion parameters.
        
        This method returns constant parameters regardless of state,
        optimized for performance by using pre-computed cache.
        
        Args:
            state: Current schedule state (ignored for fixed scheduler)
            
        Returns:
            Dictionary with fixed diffusion parameters
        """
        # Use backend implementation if available
        if self._backend_impl is not None:
            return self._backend_impl.diffusion_params(state)
        
        # Return pre-computed parameters (O(1) performance)
        return self._cached_params.copy()

