"""
Energy functional abstraction for the energy-driven control framework.

This module defines the EnergyFunctional interface, which represents the total
energy/cost of a trajectory. The energy is decomposed into:
- Task cost: primary objective (e.g., reach goal, minimize control effort)
- Constraint energy: barrier/penalty terms for constraints
- Entropy term: regularization for exploration/stochasticity

All solvers depend only on this interface, allowing easy composition of
different task objectives and constraints.
"""

from enerdynamics.core.energy.base import EnergyFunctional
from enerdynamics.core.energy.legacy import (
    LegacyEnergyFunctional,
    EnergyTerm,
    trajectory_energy_from_legacy,
)

__all__ = [
    "EnergyFunctional",
    "LegacyEnergyFunctional",
    "EnergyTerm",
    "trajectory_energy_from_legacy",
]
