"""
Posterior calibration analysis for probabilistic 3DGS reconstruction.

Computes:
  - Reliability / calibration curve (predicted std vs actual error).
  - Expected Calibration Error (ECE).
  - Coverage at various credible-interval widths.
  - Per-pixel NLL of held-out images under the predicted Gaussian posterior.
"""

from .metrics import (
    compute_reliability_curve,
    compute_ece,
    compute_coverage_curve,
    compute_pixel_nll,
    CalibrationResult,
)

__all__ = [
    "compute_reliability_curve",
    "compute_ece",
    "compute_coverage_curve",
    "compute_pixel_nll",
    "CalibrationResult",
]
