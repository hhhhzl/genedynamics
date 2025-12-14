"""
Environment modules and factories.

This package provides:
- Environment implementations (e.g., DoubleIntegratorBoxEnv)
- Factory functions for creating environments and energy functionals
"""

from enerdynamics.envs.factories import make_env, make_energy

__all__ = [
    "make_env",
    "make_energy",
]

