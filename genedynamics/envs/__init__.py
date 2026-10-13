"""
Environment modules and factories.

This package provides:
- Environment implementations (e.g., DoubleIntegratorBoxEnv)
- Factory functions for creating environments and energy functionals
"""

def make_env(name: str, **kwargs):
    """Load general factories only when requested by their caller.

    Direct NumPy/MuJoCo providers can be used without installing JAX.
    """
    from genedynamics.envs.factories import make_env as factory
    return factory(name, **kwargs)


def make_energy(env_name: str):
    from genedynamics.envs.factories import make_energy as factory
    return factory(env_name)

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

