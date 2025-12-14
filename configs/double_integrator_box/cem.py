"""
CEM (Cross-Entropy Method) configuration for double_integrator_box environment.
"""

from dataclasses import dataclass


@dataclass
class CEMArgs:
    """Configuration arguments for CEM solver."""
    seed: int = 0
    env_name: str = "double_integrator_box"
    horizon: int = 80
    dt: float = 0.1
    num_samples: int = 512
    num_iterations: int = 6
    elite_frac: float = 0.1
    init_std: float = 0.5
    min_std: float = 0.05
    action_limit: float = 1.0
    verbose: bool = False

