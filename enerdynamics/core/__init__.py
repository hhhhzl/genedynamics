"""
Core abstractions for the energy-driven control framework.

This package defines the fundamental interfaces that all components build upon:
- State, Action, Trajectory: Core data types
- DynamicsModel: System dynamics (physics or learned)
- EnergyFunctional: Task objectives and constraints
- Solver: Trajectory optimization algorithms
- Backend: Computational backends (JAX, PyTorch, Rust, etc.)

All algorithms and environments depend only on these abstractions, allowing
easy swapping of implementations and backends.
"""

# Core types
from enerdynamics.core.types import State, Action, Trajectory, StateType, ActionType, TrajectoryType

# Backend abstraction
from enerdynamics.core.backends import (
    Backend,
    JaxBackend,
    NumpyBackend,
    TorchBackend,
    get_backend,
)

# Dynamics models
from enerdynamics.core.dynamics import (
    DynamicsModel,
    DeterministicDynamicsModel,
    StochasticDynamicsModel,
    EnvDynamicsAdapter,
    DynamicsToEnvAdapter,
)

# Energy functionals
from enerdynamics.core.energy import (
    EnergyFunctional,
    LegacyEnergyFunctional,
    EnergyTerm,
    trajectory_energy_from_legacy,
)

# Solver base classes
from enerdynamics.core.solvers import (
    Solver,
    SamplingSolver,
    OptimizationSolver,
)

# Legacy utilities (for backward compatibility)
# Note: project_box and soft_box_energy are now in constraints.base for backward compatibility
# New code should use enerdynamics.core.constraints for the full constraint system
from enerdynamics.core.constraints.base import project_box, soft_box_energy
from enerdynamics.core.metrics import euclidean_metric_inv
from enerdynamics.core.integrators import langevin_step, euler_step

__all__ = [
    # Types
    "State",
    "Action",
    "Trajectory",
    "StateType",
    "ActionType",
    "TrajectoryType",
    # Backend
    "Backend",
    "JaxBackend",
    "NumpyBackend",
    "TorchBackend",
    "get_backend",
    # Dynamics
    "DynamicsModel",
    "DeterministicDynamicsModel",
    "StochasticDynamicsModel",
    "EnvDynamicsAdapter",
    "DynamicsToEnvAdapter",
    # Energy
    "EnergyFunctional",
    "LegacyEnergyFunctional",
    "EnergyTerm",
    "trajectory_energy_from_legacy",
    # Solvers
    "Solver",
    "SamplingSolver",
    "OptimizationSolver",
    # Legacy utilities
    "project_box",
    "soft_box_energy",
    "euclidean_metric_inv",
    "langevin_step",
    "euler_step",
]

