"""A separately versioned local action class preserving source force-off modes."""
from copy import deepcopy
import numpy as np
import torch
from .residual_transport import PublicResidualTransport,PublicTransportSpec,BOUNDS

VERSION='public-local-source-conditioned-transport-v1'
MAXIMUM_SCALE=.025


class PublicLocalResidualTransport(PublicResidualTransport):
    def __init__(self,response_spec=None,spec=None):
        spec=spec or PublicTransportSpec(residual_scale=MAXIMUM_SCALE)
        if not 0<=spec.residual_scale<=MAXIMUM_SCALE:raise ValueError('fixed local envelope maximum is0.025')
        super().__init__(response_spec,spec)

    def forward(self,target_features,source_features,source_commands,base_proto,phase,own_decoder_context):
        value=super().forward(target_features,source_features,source_commands,base_proto,phase,own_decoder_context)
        force=torch.where(base_proto[...,12:13]*25>.1,value[...,12:13],torch.zeros_like(value[...,12:13]))
        return torch.cat((value[...,:12],force),dim=-1)

    def checkpoint(self):
        payload=super().checkpoint();payload.update(version=VERSION,source_force_off_threshold_N=.1,scope='local action support restriction; no physical safety or uniform model-error certificate');return payload

    @classmethod
    def from_checkpoint(cls,payload,device='cpu'):
        if payload.get('version')!=VERSION or payload.get('source_force_off_threshold_N')!=.1:raise ValueError('versioned local force-off checkpoint required')
        compatible=dict(payload,version='public-source-conditioned-transport-v1');return super().from_checkpoint(compatible,device)


class PublicLocalNoiseTransport:
    """Fixed source/nominal excitation in exactly the local proto envelope."""
    def __init__(self,*,seed,scale=MAXIMUM_SCALE,chunk_steps=10):
        if not 0<=scale<=MAXIMUM_SCALE or chunk_steps!=10:raise ValueError('predeclared local10-tick envelope required')
        self.scale,self.chunk_steps=scale,chunk_steps;self.rng=np.random.default_rng(seed);self.cursor=0;self.latent=None

    def get_state(self):return deepcopy(dict(rng=self.rng.bit_generator.state,cursor=self.cursor,latent=self.latent))
    def restore(self,state):self.rng.bit_generator.state=deepcopy(state['rng']);self.cursor=state['cursor'];self.latent=deepcopy(state['latent'])

    def apply(self,raw,phase):
        raw=np.asarray(raw);base=raw.copy();base[...,:3]=np.clip(base[...,:3],-.02,.02);base[...,3:6]=np.clip(base[...,3:6],-.12,.12);base[...,6:12]=np.clip(base[...,6:12],-1,1);base[...,12]=np.clip(base[...,12],0,25)
        if self.cursor%self.chunk_steps==0:self.latent=self.rng.uniform(-1,1,base.shape)
        value=base+self.scale*BOUNDS*self.latent
        value[...,:12]=np.clip(value[...,:12],-BOUNDS[:12],BOUNDS[:12]);value[...,12]=np.where(base[...,12]>.1,np.clip(value[...,12],0,25),0.)
        self.cursor+=1;return value
