"""Reference governor: shapes a planner's reference into an execution-compatible,
constraint-admissible, rate-limited reference for the tracking policy.

Public API::

    from genedynamics.deploy.governor import (
        GovernorConfig, ReferenceGovernor, ContinuousBaseGovernor,
        CorridorAdmissibleSet, CorridorContext,
    )

Wire it into the deploy loop between follower and controller::

    gov = ContinuousBaseGovernor(CorridorAdmissibleSet(), GovernorConfig(...))
    gov.reset(state)
    ...
    intent = follower.step(t, state)
    intent = gov.govern(intent, state, CorridorContext.from_intent(intent))
    cmd    = controller.act(state, intent)
"""

from .base import (
    AdmissibleSet,
    Box,
    CorridorContext,
    Frame,
    GovernorDiagnostics,
    Halfspace,
    ReferenceGovernor,
)
from .config import GovernorConfig
from .admissible_corridor import CorridorAdmissibleSet
from .admissible_bodysdf import (
    BodyConfig,
    BodySdfAdmissibleSet,
    BodySdfContext,
)
from .continuous_base import ContinuousBaseGovernor

__all__ = [
    "GovernorConfig",
    "ReferenceGovernor",
    "AdmissibleSet",
    "ContinuousBaseGovernor",
    "CorridorAdmissibleSet",
    "BodySdfAdmissibleSet",
    "BodyConfig",
    "BodySdfContext",
    "CorridorContext",
    "GovernorDiagnostics",
    "Frame",
    "Box",
    "Halfspace",
]
