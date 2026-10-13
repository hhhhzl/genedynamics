"""Serial heterogeneous MuJoCo reference; canonical parallel provider is JAX."""
import numpy as np
from genedynamics.learning.cross_embodiment.contact import ContactState,ContactTrace,ContactChunk,ContactCommands
from genedynamics.learning.cross_embodiment.contracts import RandomRequest,mechanism_specs
from .contact import MujocoContact


class MujocoContactBatch:
    def __init__(self,config,device,mechanism):
        self.providers=tuple(MujocoContact(config,device,m) for m in mechanism_specs(mechanism))
        first=self.providers[0]
        self.config,self.model,self.sid,self.scene=config,first.model,first.sid,first.scene

    @staticmethod
    def _single(state,index):
        return ContactState((state.physics[index],),*(x[index:index+1] for x in state[1:]))

    @staticmethod
    def _join(states):
        return ContactState(tuple(d for s in states for d in s.physics),*(np.concatenate([s[i] for s in states]) for i in range(1,len(states[0]))))

    def reset_batch(self,request,*,previous=None,mask=None,initial='clear'):
        if len(request.sample_ids)!=len(self.providers):raise ValueError('contact mechanism/world batch mismatch')
        if (previous is None)!=(mask is None):raise ValueError('partial reset needs state and mask')
        if previous is not None and (len(previous.sample_id)!=len(self.providers) or np.shape(mask)!=(len(self.providers),)):raise ValueError('partial reset batch mismatch')
        return self._join([p.reset_batch(RandomRequest(request.seed,(identity,)),initial=initial,
            previous=None if previous is None else self._single(previous,i),mask=None if mask is None else mask[i:i+1])
            for i,(p,identity) in enumerate(zip(self.providers,request.sample_ids))])

    def collect_chunk(self,state,commands):
        body=None
        if isinstance(commands,ContactCommands):commands,body=commands
        if len(state.sample_id)!=len(self.providers):raise ValueError('contact mechanism/world batch mismatch')
        results=[p.collect_chunk(self._single(state,i),commands[i:i+1] if body is None else ContactCommands(commands[i:i+1],body[i:i+1])) for i,p in enumerate(self.providers)]
        trace=ContactTrace(*(np.concatenate([r.trace[i] for r in results]) for i in range(len(ContactTrace._fields))))
        return ContactChunk(self._join([r.state for r in results]),trace)

    def synchronize(self,value):pass
    def describe(self):return {'backend':'numpy','heterogeneous':True,'compiled_chunk':False,'world_models':[p.describe() for p in self.providers]}
