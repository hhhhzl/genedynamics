"""
Configuration modules for different environments and solvers.

This package provides configuration classes (Args dataclasses) for different
combinations of environments and solvers. The structure is:
    configs/<env_name>/<solver>.py

Each solver config file exports an Args dataclass with default parameters.

Usage:
    from configs.double_integrator_box.edoc import EDOCArgs
    args = EDOCArgs(seed=42, horizon=100)
    from genedynamics.solvers.single.edoc import run_edoc
    result = run_edoc(args)
"""

# Note: We don't import here to avoid circular dependencies.
# Users should import directly from the config files:
#   from configs.double_integrator_box.edoc import EDOCArgs
#   from configs.double_integrator_box.mbd import DiffusionArgs
#   etc.

__all__ = []

