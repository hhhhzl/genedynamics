"""
Environment modules and factories.

This package provides:
- Environment implementations (e.g., DoubleIntegratorBoxEnv)
- Factory functions for creating environments and energy functionals
"""

from genedynamics.envs.factories import make_env, make_energy

# Import energy registrations to register energy functionals
try:
    import genedynamics.envs.energy_registrations
except ImportError:
    pass  # Energy registry not available

# Populate the environment/energy registry from the domain subpackages
# (decorator-based registration). Broad guard: domain backends may fail to
# import on minimal installs — those names just stay absent (as before).
try:
    import genedynamics.envs.domains  
except Exception:
    pass

__all__ = [
    "make_env",
    "make_energy",
]

