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

from genedynamics.core.solvers import Solver, SamplingSolver, OptimizationSolver

# Solver implementations
from genedynamics.solvers.single.mbd.mbd import MBDSolver, run_mbd
from genedynamics.solvers.single.ebmbd.ebmbd import EBMBDSolver
from genedynamics.solvers.single.mdoc.mdoc import MDOCSolver
from genedynamics.solvers.single.cfsmbd.cfsmbd import CFSMBDSolver
from genedynamics.solvers.single.twogo.twogo import TwoGOSolver
from genedynamics.solvers.single.mppi.mppi import MPPISolver, run_mppi
from genedynamics.solvers.single.cem.cem import CEMSolver, run_cem
from genedynamics.solvers.single.dial.dial import DIALMPCSolver

__all__ = [
    # Base classes
    "Solver",
    "SamplingSolver",
    "OptimizationSolver",
    # MBD
    "MBDSolver",
    "EBMBDSolver",
    "MDOCSolver",
    "CFSMBDSolver",
    "TwoGOSolver",
    "run_mbd",
    # MPPI
    "MPPISolver",
    "run_mppi",
    # CEM
    "CEMSolver",
    "run_cem",
    # DIAL-MPC
    "DIALMPCSolver",
    # Native Torch policy solvers (loaded only when explicitly requested).
    "FADASolver",
    "Human2Sim2RobotSolver",
    "ACPSolver",
]


_NATIVE_POLICY_EXPORTS = {
    "FADASolver": ("fada", "genedynamics.solvers.single.fada.fada"),
    "Human2Sim2RobotSolver": (
        "human2sim2robot", "genedynamics.solvers.single.human2sim2robot.human2sim2robot"
    ),
    "ACPSolver": ("acp", "genedynamics.solvers.single.acp.acp"),
}


def __getattr__(name):
    """Keep optional Torch cores out of ordinary JAX planner imports."""
    if name not in _NATIVE_POLICY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    from genedynamics.core.registry.solvers import register_solver

    method, module = _NATIVE_POLICY_EXPORTS[name]
    solver = getattr(import_module(module), name)
    register_solver(f"{method}_torch", solver)
    globals()[name] = solver
    return solver

# Note: Configuration Args classes have been moved to configs/<env_name>/<solver>.py
# Import them like:
#   from configs.double_integrator_box.edoc import EDOCArgs
#   from configs.double_integrator_box.mbd import DiffusionArgs
#   etc.

