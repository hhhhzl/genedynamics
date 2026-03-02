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
from genedynamics.core.types import State, Action, Trajectory, StateType, ActionType, TrajectoryType

# Backend abstraction
from genedynamics.core.backends import (
    Backend,
    JaxBackend,
    NumpyBackend,
    TorchBackend,
    get_backend,
)

# Dynamics models
from genedynamics.core.dynamics import (
    DynamicsModel,
    DeterministicDynamicsModel,
    StochasticDynamicsModel,
    EnvDynamicsAdapter,
    DynamicsToEnvAdapter,
)

# Energy functionals
from genedynamics.core.energy import (
    EnergyFunctional,
    LegacyEnergyFunctional,
    EnergyTerm,
    trajectory_energy_from_legacy,
)

# Solver base classes
from genedynamics.core.solvers import (
    Solver,
    SamplingSolver,
    OptimizationSolver,
)

from genedynamics.core.metrics import euclidean_metric_inv
from genedynamics.core.integrators import langevin_step, euler_step
from genedynamics.core.task_spec import (
    TaskSpec,
    Legacy2DTaskSpec,
    Legacy3DTaskSpec,
    SoftZooTaskSpec,
    EnvPluginTaskSpecAdapter,
    legacy_extract_position,
    get_default_task_spec,
)

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
    "euclidean_metric_inv",
    "langevin_step",
    "euler_step",
    # TaskSpec
    "TaskSpec",
    "Legacy2DTaskSpec",
    "Legacy3DTaskSpec",
    "SoftZooTaskSpec",
    "EnvPluginTaskSpecAdapter",
    "legacy_extract_position",
    "get_default_task_spec",
]

