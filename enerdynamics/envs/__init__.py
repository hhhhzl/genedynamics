"""
Environment modules and factories.

This package provides:
- Environment implementations (e.g., DoubleIntegratorBoxEnv)
- Factory functions for creating environments and energy functionals
"""

from enerdynamics.envs.factories import make_env, make_energy

# Import energy registrations to register energy functionals
try:
    import enerdynamics.envs.energy_registrations  # noqa: F401
except ImportError:
    pass  # Energy registry not available

__all__ = [
    "make_env",
    "make_energy",
]

