"""
EDOC configuration for double_integrator_box_2d environment.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EDOCArgs:
    """Configuration arguments for EDOC solver."""
    # exp
    seed: int = 0
    np_random_seed: Optional[int] = None
    # env
    env_name: str = "double_integrator_box_2d"
    horizon: int = 80
    dt: float = 0.1
    # diffusion / ebdc
    noise_std: float = 0.05
    # optimization domain
    action_space: bool = True
    diffusion_mode: str = "reverse"  # "forward" or "reverse"
    action_diffuse_steps: int = 100
    action_beta0: float = 1e-4
    action_betaT: float = 1e-2
    action_temp: float = 0.1
    action_extra_sigma: float = 0.0
    action_stage_ratio: float = 1.0
    action_score_mode: str = "energy"  # ["reward", "energy", "learned"]
    action_nsample: int = 128
    use_antithetic: bool = True
    dyn_loss_coeff: float = 1.0
    dyn_loss_mode: str = "trajectory"  # ["terminal", "trajectory"]
    # state box
    use_state_box: bool = False
    state_low: float = -2.0
    state_high: float = 2.0
    # render / save
    save_path: Optional[str] = None
    verbose: bool = True

