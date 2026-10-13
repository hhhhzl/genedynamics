"""Legal Panda contact-state projection from shared task-space observations.

Target joints/controller history are not accepted. Phase 1 additionally needs
an actual allowed source contact at the projected pose. Unsupported states
receive status 7 and are counted before a source skill is used.
"""
from dataclasses import replace
from typing import Any,NamedTuple


class ContactSemantics(NamedTuple):
    tcp_position: Any
    tcp_rotation: Any
    tcp_twist: Any
    normal_force: Any
    phase: Any


class ContactQueryResult(NamedTuple):
    state: Any
    support: Any
    position_error: Any
    orientation_error: Any
    twist_error: Any
    contact_compatible: Any


class ContactSourceQuery:
    def __init__(self,config,*,backend="numpy",device="cpu"):
        from .contact import ContactRunner
        self.source=ContactRunner(config=replace(config,robot="panda",mobile=False),backend=backend,device=device)
        if backend=="numpy":
            from genedynamics.envs.domains.manipulation.cross_embodiment.backends.contact_query import MujocoContactQuery
            self._impl=MujocoContactQuery(self.source)
        elif backend=="jax":
            from .backends.jax.contact_query import JaxContactQuery
            self._impl=JaxContactQuery(self.source)
        else: raise ValueError("unsupported contact query provider")
    def build(self,semantics,request,*,previous=None): return self._impl.build(semantics,request,previous=previous)
    def synchronize(self,value): return self.source.synchronize(value)


def observe_contact(state,runner,xp):
    from .contact_data import contact_carry,SEMANTIC_SLICE
    carry=contact_carry(state,runner,xp); start,_=SEMANTIC_SLICE
    return ContactSemantics(carry[:,start:start+3],carry[:,start+3:start+12].reshape(-1,3,3),
        carry[:,start+12:start+18],state.measured_force,(state.measured_force>runner.config.contact_threshold).astype(xp.int32))
