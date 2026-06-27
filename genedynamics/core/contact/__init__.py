"""Generic end-effector / surface contact primitives.

Backend-agnostic protocols and concrete contact models for
manipulation tasks where a tool moves on or near a surface (wiping,
polishing, ultrasound, assembly, etc.). Composed with the SDF subsystem
(``envs.obstacles.SDFGrid3D``) via the :class:`SdfQuery` duck-type — this
package does **not** import ``envs.obstacles``.
"""

from genedynamics.core.contact.protocols import (
    ContactModel,
    ContactState,
    SdfQuery,
)
from genedynamics.core.contact.spring_damper import SpringDamperContact
from genedynamics.core.contact.rigid_friction import RigidWithFrictionContact
from genedynamics.core.contact.cbf_terms import (
    force_bound_rows,
    penetration_bound_rows,
)
from genedynamics.core.contact.elastic_foundation import (
    stiffness_field,
    winkler_force,
)

__all__ = [
    "ContactModel",
    "ContactState",
    "SdfQuery",
    "SpringDamperContact",
    "RigidWithFrictionContact",
    "force_bound_rows",
    "penetration_bound_rows",
    "stiffness_field",
    "winkler_force",
]
