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
        from genedynamics.envs.domains.manipulation.transfer_workcell.contact_runtime import ContactRunner
        self.source=ContactRunner(config=replace(config,robot="panda",mobile=False),backend=backend,device=device)
        if backend=="numpy":
            from genedynamics.envs.domains.manipulation.transfer_workcell.backends.contact_query import MujocoContactQuery
            self._impl=MujocoContactQuery(self.source)
        elif backend=="jax":
            from genedynamics.envs.domains.manipulation.transfer_workcell.backends.contact_query_jax import JaxContactQuery
            self._impl=JaxContactQuery(self.source)
        else: raise ValueError("unsupported contact query provider")
    def build(self,semantics,request,*,previous=None): return self._impl.build(semantics,request,previous=previous)
    def synchronize(self,value): return self.source.synchronize(value)
