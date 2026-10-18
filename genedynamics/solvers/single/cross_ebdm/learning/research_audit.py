"""Solver-style finite-study audit frontend; no tensor runtime at import time.

These audits have deliberately finite scope. They cannot certify a uniform
kernel bound over arbitrary learned policies or unvisited contact histories.
"""


class ResearchAudit:
    def __init__(self,*,backend="numpy",device="cpu"):
        if backend!="numpy" or device!="cpu":
            raise NotImplementedError("finite research audit currently qualifies numpy/cpu only")
        from .backends.numpy.research_audit import NumpyResearchAudit
        self._impl=NumpyResearchAudit()
        self.backend,self.device=backend,device

    def separation(self,signatures,*,noise_bound,model_bias_bound):
        return self._impl.separation(signatures,noise_bound,model_bias_bound)

    def cost_certificate(self,actual,predicted,*,alpha=.05):
        return self._impl.cost_certificate(actual,predicted,alpha)

    def paired_improvement(self,actual,*,alpha=.05):
        """Direct observed bounded-cost comparison, conditional on calibration."""
        return self._impl.paired_improvement(actual,alpha)

    def mode_law(self,costs,source_weights,feasible,*,temperature=.2):
        return self._impl.mode_law(costs,source_weights,feasible,temperature)


def realized_signature(trace,phases,config):
    """Bounded, phase-conditioned *physical* features at an artifact boundary."""
    import numpy as np
    labels=np.repeat(np.asarray(phases).ravel(),10)[:trace["valid"].shape[1]]
    valid=trace["valid"][0]; features=[]
    for phase in range(4):
        mask=valid&(labels==phase)
        if not mask.any(): features.extend([0.]*6);continue
        xy=(trace["tcp_position"][0,mask,:2]-np.asarray(config.goal)[:2])/.02
        features.extend([1.,mask.sum()*config.control_dt/config.timeout_seconds,*np.clip(xy.mean(0),-1.,1.),
            float(np.clip(trace["normal_force"][0,mask].mean()/10.,-1.,1.)),
            float(np.clip(trace["lateral_force_peak"][0,mask].max()/config.lateral_force_limit,0.,1.))])
    return np.asarray(features).tolist()
