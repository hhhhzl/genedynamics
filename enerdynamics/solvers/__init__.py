"""
Solver implementations for the energy-driven control framework.

This package contains implementations of various solvers that all follow
the Solver interface defined in core.solver_base:

- EDOC: Energy-Driven Optimal Control using diffusion + A-MCSA + ADM
- MBD: Multi-scale Barrier Diffusion
- MPPI: Model Predictive Path Integral
- iLQR: Iterative Linear Quadratic Regulator
- Flow Matching: Learned dynamics + energy minimization

All solvers share the same interface:
    solver = Solver(dynamics, energy, backend)
    trajectory = solver.solve(x0, horizon)
"""

from enerdynamics.core.solvers import Solver, SamplingSolver, OptimizationSolver

# Solver implementations
from enerdynamics.solvers.edoc import (
    EDOCSolver,
    EDOCPlanner,
    run_edoc,
)
from enerdynamics.solvers.mbd import MBDSolver, run_mbd
from enerdynamics.solvers.mppi import MPPISolver, run_mppi
from enerdynamics.solvers.cem import CEMSolver, run_cem
from enerdynamics.solvers.ilqr import iLQRSolver

__all__ = [
    # Base classes
    "Solver",
    "SamplingSolver",
    "OptimizationSolver",
    # EDOC
    "EDOCSolver",
    "EDOCPlanner",
    "run_edoc",
    # MBD
    "MBDSolver",
    "run_mbd",
    # MPPI
    "MPPISolver",
    "run_mppi",
    # CEM
    "CEMSolver",
    "run_cem",
    # Other implementations
    "iLQRSolver",
]

# Note: Configuration Args classes have been moved to configs/<env_name>/<solver>.py
# Import them like:
#   from configs.double_integrator_box.edoc import EDOCArgs
#   from configs.double_integrator_box.mbd import DiffusionArgs
#   etc.

