"""Safety filters for the deploy pipeline.

Each filter implements the :class:`SafetyFilter` protocol from
:mod:`genedynamics.deploy.interfaces.safety` and projects a candidate
:class:`ControlCommand` onto the safe set defined by joint / torque limits,
self-collision avoidance, or a CBF.

* :class:`BaseSafetyFilter`     — abstract scaffolding
* :class:`JointLimitFilter`     — clip ``joint_pos`` to ``RobotSpec.joint_range``
* :class:`TorqueLimitFilter`    — clip ``joint_torque`` to ``actuator_forcerange``
* :class:`SelfCollisionFilter`  — best-effort capsule check via ``RobotModel``
* :class:`CBFFilter`            — bridges to ``core/constraints/convexify/cbf``
* :class:`CompositeSafetyFilter`— ordered chain of filters

Filters are pure with respect to ``RobotState`` (read-only) and
``ControlCommand`` (returned modified). They never call back into the IO.
"""

from genedynamics.deploy.safety.base import BaseSafetyFilter
from genedynamics.deploy.safety.composite import CompositeSafetyFilter
from genedynamics.deploy.safety.joint_limit import JointLimitFilter
from genedynamics.deploy.safety.self_collision import SelfCollisionFilter
from genedynamics.deploy.safety.torque_limit import TorqueLimitFilter

# CBF filter is optional (depends on core/constraints/convexify/cbf availability).
try:
    from genedynamics.deploy.safety.cbf_filter import CBFFilter
    _HAS_CBF = True
except ImportError:  # pragma: no cover
    _HAS_CBF = False
    CBFFilter = None  # type: ignore[assignment]

__all__ = [
    "BaseSafetyFilter",
    "CompositeSafetyFilter",
    "JointLimitFilter",
    "SelfCollisionFilter",
    "TorqueLimitFilter",
]
if _HAS_CBF:
    __all__.append("CBFFilter")
