"""Fixed ten-tick local excitation in the actual T1/T2 action Box."""
from copy import deepcopy
import numpy as np
from genedynamics.tasks.transfer.controllers.guide_feedback import PublicGuideFeedbackPolicy
from genedynamics.tasks.transfer.controllers.obstacle_feedback import PublicObstacleFeedbackPolicy


class PublicArmTaskNoiseActor:
    VERSION='public-T1-T2-actualrequest-local-noise-v1'
    def __init__(self,spec,*,mode,seed):
        if spec.alias not in {'T1','T2'}:raise ValueError('explicit T1/T2 excitation only')
        self.spec=spec;self.mode=mode;self.seed=seed;self.base=(PublicObstacleFeedbackPolicy(spec,modes=[mode]) if spec.alias=='T1' else PublicGuideFeedbackPolicy(spec,modes=[mode],lead_xy_m=.008,position_admittance=True));self.reset()
    def reset(self):self.base.reset();self.rng=np.random.default_rng(self.seed);self.latent=None;self.tick=-1;self.cached=None
    def get_state(self):return deepcopy(dict(version=self.VERSION,base=self.base.get_state(),rng=self.rng.bit_generator.state,latent=self.latent,tick=self.tick,cached=self.cached))
    def restore(self,s):
        if s['version']!=self.VERSION:raise ValueError('noisebinding differs')
        self.base.restore(s['base']);self.rng.bit_generator.state=deepcopy(s['rng']);self.latent=deepcopy(s['latent']);self.tick=s['tick'];self.cached=deepcopy(s['cached'])
    def act(self,public):
        tick=int(np.rint(np.asarray(public['elapsed_seconds']).item()/self.spec.config.control_dt))
        if tick==self.tick:return self.cached.copy()
        if tick!=self.tick+1:raise ValueError('all causalpublicticks required')
        raw=self.base.act(public);bounds=np.asarray([.12]*3+[.6]*3+[1.]*6+[0.],np.float32) if self.spec.alias=='T1' else np.asarray([.08]*3+[.4]*3+[1.]*6+[25.],np.float32)
        if tick%10==0:self.latent=self.rng.uniform(-1,1,raw.shape)
        value=np.clip(raw+.025*bounds*self.latent,-bounds,bounds);value[:,12]=np.where(raw[:,12]>.1,np.clip(value[:,12],0,bounds[12]),0.);self.cached=value.astype(np.float32);self.tick=tick;return self.cached.copy()
