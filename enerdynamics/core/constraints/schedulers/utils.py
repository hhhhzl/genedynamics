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
        
        Args:
            k: Diffusion step index
            snr_min: Minimum SNR value (for normalization)
            snr_max: Maximum SNR value (for normalization)
            
        Returns:
            Normalized progress t_k in [0, 1]
        """
        snr_k = self.get_snr(k)
        
        # Handle edge cases
        if snr_k <= 0.0:
            return 0.0
        if not np.isfinite(snr_k):
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
        # Filter out inf values
        finite_snrs = [s for s in all_snrs if np.isfinite(s)]
        if not finite_snrs:
            return (0.0, 1.0)
        return (float(min(finite_snrs)), float(max(finite_snrs)))

