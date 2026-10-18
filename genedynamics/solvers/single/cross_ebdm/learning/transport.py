"""Thin random residual-transport frontend, independent of the physics provider."""
from dataclasses import dataclass
from typing import Any,NamedTuple


@dataclass(frozen=True)
class TransportSpec:
    horizon: int=10
    residual_scale: float=.15
    learning_rate: float=.0003
    change_weight: float=.01
    risk_multiplier: float=.01
    tail_fraction: float=.2
    microbatch_size: int=16
    motion_residual_floor: float=0.
    task_weight: float=0.

    def __post_init__(self):
        import math
        if self.horizon<1 or self.microbatch_size<1 or not 0<self.tail_fraction<=1:
            raise ValueError("invalid transport horizon/microbatch/tail")
        if not 0<=self.motion_residual_floor<=1: raise ValueError("invalid motion residual floor")
        if any(not math.isfinite(x) or x<0 for x in (self.residual_scale,self.learning_rate,self.change_weight,self.risk_multiplier,self.task_weight)):
            raise ValueError("invalid transport coefficients")


class TransportBatch(NamedTuple):
    source_carry: Any
    target_carry: Any
    source_commands: Any
    context: Any
    valid: Any
    source_body: Any=None
    target_body: Any=None
    task_reference: Any=None  # [B,13]: phase waypoint xyz, rotation[9], normal force


class StochasticTransport:
    def __init__(self,response,*,spec=None,backend="jax",device="cpu"):
        self.spec=spec or TransportSpec()
        if backend=="jax":
            from .backends.jax.transport import JaxTransport
            provider=JaxTransport
        elif backend=="torch":
            from .backends.torch.transport import TorchTransport
            provider=TorchTransport
        else: raise NotImplementedError("transport provider supports JAX or Torch")
        self._impl=provider(response,self.spec,device)

    def initialize(self,seed=0): return self._impl.initialize(seed)
    def apply(self,parameters,batch,noise): return self._impl.apply(parameters,batch,noise)
    def train_step(self,state,batch,noise): return self._impl.train_step(state,batch,noise)
    def train_chunk(self,state,batches,noises): return self._impl.train_chunk(state,batches,noises)
    def synchronize(self,value): return self._impl.model.synchronize(value)

    def save(self,path,parameters):
        from dataclasses import asdict
        self._impl.model.save(path,parameters,metadata={"artifact_role":"stochastic_transport","transport_spec":asdict(self.spec)})

    @classmethod
    def load(cls,response,path,*,backend="jax",device="cpu"):
        from .models import LearnedModel
        model,parameters,metadata=LearnedModel.load(path,backend=backend,device=device)
        if metadata.get("artifact_role")!="stochastic_transport": raise ValueError("expected transport artifact")
        instance=cls(response,spec=TransportSpec(**metadata["transport_spec"]),backend=backend,device=device)
        if instance._impl.model.spec!=model.spec: raise ValueError("transport/response layout mismatch")
        return instance,parameters
