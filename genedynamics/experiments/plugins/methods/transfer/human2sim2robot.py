"""Public held-object trajectory binding of the original recurrent PPO actor.

Robot source trajectories replace human assets explicitly. Reference-clock
updates match ObjectTrajectoryTaskEnv; deployment never holds a physics task.
"""
from copy import deepcopy
import hashlib
import numpy as np
import torch
from genedynamics.envs.adapters.transfer_task import FLAT_FIELDS
from genedynamics.solvers.single.cross_ebdm.learning.response import PUBLIC_FIELDS
from genedynamics.solvers.single.human2sim2robot import Human2Sim2RobotSolver
from genedynamics.solvers.single.human2sim2robot.backends.native.network import NetworkConfig,MlpConfig,RnnConfig

VERSION='public-held-object-native-PPO-v1'
def network_config(value):
    return NetworkConfig(**dict(value,mlp=MlpConfig(**value['mlp']),rnn=RnnConfig(**value['rnn']) if value['rnn'] else None))
def tracking_distance(public,reference):
    offsets=np.eye(3,dtype=np.float32)*.2
    actual=np.einsum('bij,kj->bki',public['tcp_rotation'],offsets)+public['tcp_position'][:,None]
    desired=np.einsum('bij,kj->bki',reference[None,:3,:3],offsets)+reference[None,None,:3,3]
    return np.linalg.norm(actual-desired,axis=-1).mean(axis=-1).astype(np.float32)

class PublicObjectTrajectoryPPOActor:
    VERSION=VERSION
    def __init__(self,spec,packet,*,mode,seed=0,device='cpu'):
        meta=packet['public_task_adapter']
        if meta['version']!=VERSION or meta['task']!=spec.alias or meta['body']!=spec.config.robot or meta['mode']!=mode or tuple(meta['flat_fields'])!=FLAT_FIELDS:raise ValueError('explicit task/body/mode/public chart binding required')
        self.spec,self.mode,self.seed=spec,int(mode),int(seed);self.meta=deepcopy(meta);self.demo=np.asarray(meta['demonstration'],np.float32);self.device=torch.device(device)
        if self.demo.shape[1:]!=(4,4) or hashlib.sha256(self.demo.tobytes()).hexdigest()!=meta['demonstration_sha256']:raise ValueError('frozen robot source trajectory changed')
        self.solver=Human2Sim2RobotSolver(network_config=network_config(meta['network_config']),observation_shape=meta['observation_shape'],action_low=meta['action_low'],action_high=meta['action_high'],device=device,normalize_input=True,normalize_value=True)
        self.solver.restore(packet['native_policy']);self.irreversible_network_calls=0;self.reset()
    def reset(self):
        self.solver.reset(1);self.tick=-1;self.reference_index=0.;self.cached=None
        generator=torch.Generator(device=self.device);generator.manual_seed(self.seed);self.rng=generator.get_state()
    def get_state(self):
        return deepcopy(dict(version=VERSION,tick=self.tick,reference_index=self.reference_index,cached=self.cached,rng=self.rng,states=self.solver.states,batch_size=self.solver._backend_impl.batch_size))
    def restore(self,state):
        if state['version']!=VERSION:raise ValueError('public PPO memory version differs')
        self.tick=state['tick'];self.reference_index=state['reference_index'];self.cached=deepcopy(state['cached']);self.rng=state['rng'].clone();self.solver._backend_impl.states=deepcopy(state['states']);self.solver._backend_impl.batch_size=state['batch_size']
    def observation(self,public):
        reference=self.demo[min(int(self.reference_index),len(self.demo)-1)]
        return np.concatenate((*[np.asarray(public[k]).reshape(1,-1) for k in FLAT_FIELDS],reference[None,:3,3],reference[None,:3,:3].reshape(1,9)),axis=-1).astype(np.float32)
    def act(self,public):
        if set(public)!=PUBLIC_FIELDS or len(public['tcp_position'])!=1:raise ValueError('one exact public world required')
        tick=int(np.rint(np.asarray(public['elapsed_seconds']).item()/self.spec.config.control_dt))
        if tick==self.tick:return self.cached.copy()
        if tick!=self.tick+1:raise ValueError('every public tick and explicit cold reset required')
        if tick:
            old_reference=self.demo[min(int(self.reference_index),len(self.demo)-1)]
            self.reference_index+=0. if tracking_distance(public,old_reference)[0]>.2 else self.spec.config.control_dt/self.meta['demo_dt']
        # Native inference samples even when deterministic mu is selected.
        # Each public actor owns that RNG stream; unrelated actors cannot alter
        # continuation or receive this actor's random state.
        devices=[self.device.index or 0] if self.device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            if devices:torch.cuda.set_rng_state(self.rng,self.device)
            else:torch.set_rng_state(self.rng)
            self.cached=self.solver.act(self.observation(public)).detach().cpu().numpy().astype(np.float32);self.irreversible_network_calls+=1
            self.rng=torch.cuda.get_rng_state(self.device) if devices else torch.get_rng_state()
        self.tick=tick;return self.cached.copy()
