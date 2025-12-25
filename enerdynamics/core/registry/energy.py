"""
Energy functional registry for energy function factories.

This registry manages energy functional factory functions. Each registered
entry should be a callable that returns a LegacyEnergyFunctional instance.
"""

from typing import Callable
from enerdynamics.core.energy import LegacyEnergyFunctional
from enerdynamics.core.registry.base import BaseRegistry

# Type hint for energy factory functions
EnergyFactory = Callable[..., LegacyEnergyFunctional]

# Global energy registry instance
_energy_registry = BaseRegistry[LegacyEnergyFunctional]("enerdynamics.energy")


def register_energy(
    name: str,
    factory: EnergyFactory
) -> None:
    """
    Register an energy functional factory.
    
    Args:
        name: Energy name (e.g., "double_integrator_box")
        factory: Factory function that creates a LegacyEnergyFunctional
    
    Examples:
        >>> def make_my_energy():
        ...     return LegacyEnergyFunctional({...})
        >>> register_energy("my_energy", make_my_energy)
    """
    _energy_registry.register(name, LegacyEnergyFunctional, factory=factory)


def get_energy_registry() -> BaseRegistry[LegacyEnergyFunctional]:
    """
    Get the global energy registry.
    
    Returns:
        Energy registry instance
    """
    return _energy_registry

