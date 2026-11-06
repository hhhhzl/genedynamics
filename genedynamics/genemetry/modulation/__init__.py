"""
Task modulators for constraint geometry adaptation.

Modulators are backend-agnostic (basic arithmetic only) and do not
require backend dispatch.
"""

from importlib import import_module

from genedynamics.genemetry.modulation.null import NullModulator
from genedynamics.genemetry.modulation.probe import ProbeModulator

__all__ = ["NullModulator", "ProbeModulator", "GoalDirectionJax"]


def __getattr__(name):
    if name != "GoalDirectionJax":
        raise AttributeError(name)
    value = getattr(
        import_module(".goal_direction", __name__),
        name,
    )
    globals()[name] = value
    return value
