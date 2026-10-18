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
    "Play2PerfectSolver",
    "ACPSolver",
]


_NATIVE_POLICY_EXPORTS = {
    "FADASolver": ("fada", "genedynamics.solvers.single.fada.fada"),
    "Human2Sim2RobotSolver": (
        "human2sim2robot", "genedynamics.solvers.single.human2sim2robot.human2sim2robot"
    ),
    "Play2PerfectSolver": (
        "play2perfect", "genedynamics.solvers.single.play2perfect.play2perfect"
    ),
    "ACPSolver": ("acp", "genedynamics.solvers.single.acp.acp"),
}

_LAZY_EXPORTS = {
    "Solver": ("genedynamics.core.solvers", "Solver"),
    "SamplingSolver": ("genedynamics.core.solvers", "SamplingSolver"),
    "OptimizationSolver": ("genedynamics.core.solvers", "OptimizationSolver"),
    "MBDSolver": ("genedynamics.solvers.single.mbd.mbd", "MBDSolver"),
    "run_mbd": ("genedynamics.solvers.single.mbd.mbd", "run_mbd"),
    "EBMBDSolver": ("genedynamics.solvers.single.ebmbd.ebmbd", "EBMBDSolver"),
    "MDOCSolver": ("genedynamics.solvers.single.mdoc.mdoc", "MDOCSolver"),
    "CFSMBDSolver": ("genedynamics.solvers.single.cfsmbd.cfsmbd", "CFSMBDSolver"),
    "TwoGOSolver": ("genedynamics.solvers.single.twogo.twogo", "TwoGOSolver"),
    "MPPISolver": ("genedynamics.solvers.single.mppi.mppi", "MPPISolver"),
    "run_mppi": ("genedynamics.solvers.single.mppi.mppi", "run_mppi"),
    "CEMSolver": ("genedynamics.solvers.single.cem.cem", "CEMSolver"),
    "run_cem": ("genedynamics.solvers.single.cem.cem", "run_cem"),
    "DIALMPCSolver": ("genedynamics.solvers.single.dial.dial", "DIALMPCSolver"),
}


def __getattr__(name):
    """Keep optional Torch cores out of ordinary JAX planner imports."""
    from importlib import import_module

    if name in _LAZY_EXPORTS:
        module, attribute = _LAZY_EXPORTS[name]
        value = getattr(import_module(module), attribute)
        globals()[name] = value
        return value
    if name not in _NATIVE_POLICY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
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

