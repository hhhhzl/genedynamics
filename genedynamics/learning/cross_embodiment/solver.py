"""Frozen inference orchestrator, following solver/backend interface separation."""
from typing import Any,NamedTuple,Protocol


class PolicyChunk(NamedTuple):
    state: Any
    source_seed: Any
    trace: Any
    support: Any
    support_terminal: Any
    commands: Any


class PolicyBackend(Protocol):
    def run_chunk(self,state,source_seed,modes,prior_noise,transport_noise,context) -> PolicyChunk: ...
    def synchronize(self,value): ...


class CrossEmbodimentSolver:
    """Inputs/noise are already device arrays; no target truth is accepted.

    A chunk contains K planning windows, each with H physical control steps.
    The JAX provider fuses source projection, prior, transport and execution
    under an outer scan. Offline NumPy artifact scripts are separate from this
    canonical device path.
    """
    def __init__(self,runner,query,prior,*,transport=None,parameters=None,backend="jax"):
        if backend!="jax": raise NotImplementedError("fused policy/execution provider currently JAX only")
        if hasattr(runner,"config"):
            from .backends.jax.contact_solver import JaxContactPolicy
            self._impl=JaxContactPolicy(runner,query,prior,transport,parameters)
        else:
            from .backends.jax.solver import JaxPolicy
            self._impl=JaxPolicy(runner,query,prior,transport,parameters)
    def initial_source_seed(self,batch): return self._impl.initial_source_seed(batch)
    def run_chunk(self,state,source_seed,modes,prior_noise,transport_noise,context,*,body_commands=None):
        if body_commands is not None:
            if not hasattr(self._impl,"source_body"): raise ValueError("body commands require a contact/mobile policy provider")
            return self._impl.run_chunk(state,source_seed,modes,prior_noise,transport_noise,context,body_commands=body_commands)
        return self._impl.run_chunk(state,source_seed,modes,prior_noise,transport_noise,context)
    def synchronize(self,value): return self._impl.synchronize(value)
