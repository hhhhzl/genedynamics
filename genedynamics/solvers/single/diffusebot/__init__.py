"""DiffuseBot — physics-augmented generative diffusion co-design.

Faithful reproduction of Wang et al. (NeurIPS 2023) against /workspace/DiffuseBot.
Thin solver shell + JAX-MPM backend (mirrors the SHAC / MRMFMBD packaging).
"""

from .protocols import DiffuseBotConfig
from .diffusebot import DiffuseBotSolver
from .backends import DiffuseBotBackendJax

__all__ = ["DiffuseBotConfig", "DiffuseBotSolver", "DiffuseBotBackendJax"]
