"""
Legacy energy functional support for backward compatibility.

This module provides support for the old point-wise energy evaluation interface
that evaluates energy at individual (state, action) pairs rather than full trajectories.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Tuple

import numpy as np

from genedynamics.core.dynamics.base import DynamicsModel
from genedynamics.core.energy.base import EnergyFunctional
from genedynamics.core.types import Trajectory

Array = np.ndarray


@dataclass
class EnergyTerm:
    """
    Legacy energy term for point-wise energy evaluation.
    
    This is kept for backward compatibility with existing code that evaluates
    energy at individual (state, action) pairs rather than full trajectories.
    """
    fn: Callable[[Array, Array, Dict[str, Any]], Array]
    weight: float = 1.0

    def __call__(self, x: Array, u: Array, info: Dict[str, Any] | None):
        return self.weight * self.fn(x, u, info)


class LegacyEnergyFunctional:
    """
    Legacy energy functional for point-wise evaluation.
    
    This class evaluates energy at individual (state, action) pairs.
    It can be wrapped to create a trajectory-based EnergyFunctional.
    """
    def __init__(self, terms: Dict[str, EnergyTerm]):
        self.terms = dict(terms)
        self._term_items: Tuple[Tuple[str, EnergyTerm], ...] = tuple(self.terms.items())

    def __call__(self, x: Array, u: Array, info: Dict[str, Any] = None) -> Array:
        return self.compute(x, u, info)

    def compute(self, x: Array, u: Array, info: Dict[str, Any] | None = None) -> Array:
        total = 0.0
        for _, term in self._term_items:
            total = total + term(x, u, info)
        return total

    def compute_terms(self, x: Array, u: Array, info: Dict[str, Any] | None = None) -> Iterable[Tuple[str, Array]]:
        for name, term in self._term_items:
            yield name, term(x, u, info)

    def breakdown(self, x: Array, u: Array, info: Dict[str, Any] = None):
        out = {}
        info_local = info or {}
        for name, term in self.terms.items():
            out[name] = float(term(x, u, info_local))
        return out


def trajectory_energy_from_legacy(legacy_energy: LegacyEnergyFunctional) -> EnergyFunctional:
    """
    Convert a legacy point-wise energy functional to a trajectory-based one.
    
    This helper function sums the legacy energy over all (state, action) pairs
    in the trajectory.
    
    Args:
        legacy_energy: Legacy energy functional
        
    Returns:
        Trajectory-based energy functional
    """
    class WrappedEnergy(EnergyFunctional):
        def task_cost(self, traj: Trajectory) -> float:
            total = 0.0
            for i, (x, u) in enumerate(zip(traj.states[:-1], traj.actions)):
                # Extract data from State/Action protocols
                x_data = x.data if hasattr(x, 'data') else x
                u_data = u.data if hasattr(u, 'data') else u
                total += float(legacy_energy.compute(x_data, u_data, traj.info))
            return total
        
        def constraint_energy(self, traj: Trajectory) -> float:
            # For legacy energy, constraints are mixed with task cost
            # This is a simple wrapper - more sophisticated decomposition
            # would require refactoring the legacy energy terms
            return 0.0

    return WrappedEnergy()


class EnergyToLegacyAdapter:
    """
    Converts a trajectory-based EnergyFunctional to point-wise LegacyEnergyFunctional.

    Used by MBD, EB-MBD, MPPI, MDOC, CEM, CFSMBD solvers whose backends expect
    the legacy (x, u) -> scalar interface.
    """

    def __init__(self, energy: EnergyFunctional, dynamics: DynamicsModel):
        self._energy = energy
        self._dynamics = dynamics
        self.legacy_energy = self._build_legacy()

    def _build_legacy(self) -> LegacyEnergyFunctional:
        def step_energy(x: Array, u: Array, info: Dict[str, Any] | None) -> Array:
            info = info or {}
            x_next = self._dynamics.step(x, u)
            traj = Trajectory(states=[x, x_next], actions=[u], info=info)
            return float(self._energy.total_energy(traj))

        return LegacyEnergyFunctional({"total": EnergyTerm(step_energy, 1.0)})
