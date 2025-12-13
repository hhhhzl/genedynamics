"""
MBD (Multi-scale Barrier Diffusion) configuration for double_integrator_box_2d environment.
"""

from dataclasses import dataclass


@dataclass
class DiffusionArgs:
    """Configuration arguments for MBD diffusion solver."""
    seed: int = 0
    env_name: str = "double_integrator_box_2d"
    horizon: int = 80
    dt: float = 0.1
    Nsample: int = 2048
    Ndiffuse: int = 100
    temp_sample: float = 0.1
    beta0: float = 1e-4
    betaT: float = 1e-2
    action_limit: float = 1.0
    verbose: bool = True

