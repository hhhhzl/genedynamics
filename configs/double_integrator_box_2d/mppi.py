"""
MPPI (Model Predictive Path Integral) configuration for double_integrator_box_2d environment.
"""

from dataclasses import dataclass


@dataclass
class MPPIArgs:
    """Configuration arguments for MPPI solver."""
    seed: int = 0
    env_name: str = "double_integrator_box_2d"
    horizon: int = 80
    dt: float = 0.1
    num_samples: int = 512
    num_iterations: int = 6
    noise_sigma: float = 0.3
    lambda_: float = 1.0
    action_limit: float = 1.0
    verbose: bool = False

