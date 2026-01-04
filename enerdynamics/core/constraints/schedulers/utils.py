"""
Utility classes for schedulers.

Provides shared utilities like DiffusionNoiseSchedule for computing
diffusion progress and SNR-based scheduling.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DiffusionNoiseSchedule:
    """
    Diffusion noise schedule (alpha_k, alpha_bar_k).
    
    Computes and stores diffusion schedule parameters for SNR-based
    progress calculation used in dual-control schedulers.
    
    Attributes:
        alphas: [K] array of alpha_k values (1 - beta_k)
        alphas_bar: [K] array of cumulative products (alpha_bar_k = prod(alpha_1..alpha_k))
        betas: [K] array of beta_k values (noise schedule)
    """
    alphas: np.ndarray  # [K] array of alpha_k values
    alphas_bar: np.ndarray  # [K] array of cumulative products
    betas: np.ndarray  # [K] array of beta_k values
    
    @classmethod
    def from_betas(cls, betas: np.ndarray) -> "DiffusionNoiseSchedule":
        """
        Create diffusion schedule from beta schedule.
        
        Args:
            betas: [K] array of beta_k values (noise schedule)
            
        Returns:
            DiffusionNoiseSchedule instance
        """
        betas = np.asarray(betas, dtype=np.float32)
        alphas = 1.0 - betas
        alphas_bar = np.cumprod(alphas, axis=0)
        return cls(alphas=alphas, alphas_bar=alphas_bar, betas=betas)
    
    def get_snr(self, k: int) -> float:
        """
        Get signal-to-noise ratio (SNR) at diffusion step k.
        
        SNR_k = alpha_bar_k / (1 - alpha_bar_k)
        
        Args:
            k: Diffusion step index (0-indexed)
            
        Returns:
            SNR value at step k
        """
        if k < 0 or k >= len(self.alphas_bar):
            raise IndexError(f"Step k={k} out of range [0, {len(self.alphas_bar)})")
        
        alpha_bar_k = self.alphas_bar[k]
        # Avoid division by zero
        if alpha_bar_k >= 1.0:
            return float('inf')
        if alpha_bar_k <= 0.0:
            return 0.0
        
        return float(alpha_bar_k / (1.0 - alpha_bar_k))
    
    def get_progress(self, k: int, snr_min: float, snr_max: float) -> float:
        """
        Get normalized progress t_k in [0, 1] from SNR.
        
        t_k = (log(SNR_k) - log(SNR_min)) / (log(SNR_max) - log(SNR_min))
        
        Larger t_k indicates later denoising (lower noise, closer to final solution).
        In reverse diffusion: t_k = 1.0 for initial high noise, t_k = 0.0 for final low noise.
        
        Args:
            k: Diffusion step index
            snr_min: Minimum SNR value (for normalization)
            snr_max: Maximum SNR value (for normalization, may be adjusted for inf values)
            
        Returns:
            Normalized progress t_k in [0, 1]
        """
        snr_k = self.get_snr(k)
        
        # Handle edge cases
        if snr_k <= 0.0:
            return 0.0
        
        # Handle infinite SNR (alpha_bar_k >= 1.0, which means very high noise)
        # In reverse diffusion, this represents the initial step with high noise
        # t_k should be 1.0 (initial high noise), not 0.0
        if not np.isfinite(snr_k):
            # If snr_max was adjusted for inf values (see compute_snr_bounds),
            # we can compute t_k using the adjusted snr_max
            # Otherwise, return 1.0 directly (initial high noise state)
            if snr_max > 1e5:  # snr_max was likely adjusted for inf values
                # Use the adjusted snr_max to compute t_k
                # This ensures smooth transition from inf to finite SNR values
                log_snr_min = np.log(snr_min)
                log_snr_max = np.log(snr_max)
                if log_snr_max > log_snr_min:
                    # For inf SNR, t_k should approach 1.0 (initial high noise)
                    # Use a value slightly less than 1.0 to allow for smooth interpolation
                    return 0.999
            return 1.0
        
        log_snr_k = np.log(snr_k)
        log_snr_min = np.log(snr_min)
        log_snr_max = np.log(snr_max)
        
        if log_snr_max <= log_snr_min:
            return 0.0
        
        t_k = (log_snr_k - log_snr_min) / (log_snr_max - log_snr_min)
        return float(np.clip(t_k, 0.0, 1.0))
    
    def compute_snr_bounds(self) -> tuple[float, float]:
        """
        Compute SNR bounds from the schedule.
        
        Returns:
            Tuple of (snr_min, snr_max)
        """
        all_snrs = [self.get_snr(k) for k in range(len(self.alphas_bar))]
        # Separate finite and infinite values
        finite_snrs = [s for s in all_snrs if np.isfinite(s)]
        has_inf = any(not np.isfinite(s) for s in all_snrs)
        
        if not finite_snrs:
            return (0.0, 1.0)
        
        snr_min = float(min(finite_snrs))
        snr_max = float(max(finite_snrs))
        
        # If there are inf values (alpha_bar_k >= 1.0), we need to set snr_max
        # to a large finite value for proper normalization in get_progress.
        # Use a value that's much larger than the max finite SNR to ensure
        # proper interpolation when computing t_k.
        if has_inf:
            # Set snr_max to be 100x the max finite SNR, or use a fixed large value
            # This ensures that when we compute t_k for inf SNR, we get t_k = 1.0
            # (which represents initial high noise in reverse diffusion)
            snr_max = max(snr_max * 100.0, 1e6)
        
        return (snr_min, snr_max)

