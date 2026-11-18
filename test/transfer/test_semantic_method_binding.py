import numpy as np

from genedynamics.solvers.single.cross_ebdm.learning.response import PUBLIC_FIELDS
from genedynamics.tasks.transfer.controllers.guide_contact import (
    PublicCornerGuidePolicy, adapt_corner_public_policy,
    bind_corner_public_planner,
)
from genedynamics.tasks.transfer.specs import task_spec


def test_corner_method_binding_installs_public_planner_and_filters_schema():
    class Actor:
        def __init__(self):
            self.base = object()
            self.keys = None

        def act(self, public):
            self.keys = set(public)
            return np.zeros((1, 13), dtype=np.float32)

        def reset(self):
            return None

    actor = Actor()
    execution = bind_corner_public_planner(
        actor, task_spec("corner_guided_seating", robot="xarm7"), mode=0,
    )
    assert isinstance(actor.base, PublicCornerGuidePolicy)
    assert actor.spec.name == "corner_guided_seating"
    public = {key: np.zeros((1, 1), dtype=np.float32) for key in PUBLIC_FIELDS}
    public["semantic_only"] = np.ones((1, 1), dtype=np.float32)
    np.testing.assert_array_equal(
        execution.act(public), np.zeros((1, 13), dtype=np.float32),
    )
    assert actor.keys == PUBLIC_FIELDS


def test_corner_parameter_adapter_retains_only_declared_t2_coordinates():
    class Actor:
        AUTHORITY_GROWTH_PER_CONTROL = .039
        NOMINAL_LEAD_M = .0095
        MAXIMUM_CONTACT_SPEED_MPS = .036

    execution = adapt_corner_public_policy(
        Actor(), task_spec("corner_guided_seating", robot="xarm7"), mode=0,
    )
    state = execution.get_state()["policy"]
    assert state["authority_growth_per_control"] == .039
    assert state["nominal_lead_m"] == .0095
    assert state["maximum_contact_speed_mps"] == .036
