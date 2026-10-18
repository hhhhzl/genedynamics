"""Explicit T4 binding with independent public memory and actual mobile identity."""
import torch
from genedynamics.solvers.single.response_mpc.contact import PublicResponseMPCActor
from genedynamics.tasks.transfer.controllers.mating_feedback import (
    PublicLivenessMatingFeedbackPolicy,
    PublicMatingFeedbackPolicy,
)
from .dense import ProtoInterceptor, PublicDenseTransportActor


class PublicMobileDenseTransportActor(PublicDenseTransportActor):
    """T3 source queries, T4 target features, unchanged actual Go2+Z1 executor.

    This controller supplies arm skill commands under the shared stance WBC.
    It adds neither a gait/navigation policy nor a coupled-body claim.
    """
    BASE_POLICY=PublicMatingFeedbackPolicy

    def __init__(self,spec,policy,source_bank,feedback,*,prior_path=None,mode=0,seed=0,device='cpu',maximum_query_distance=5.):
        if spec.alias!='T4' or spec.config.robot!='go2_z1':raise ValueError('actual T4 mobile binding required')
        if source_bank.task!='T3' or source_bank.spec!=policy.response_spec:raise ValueError('T4 uses declared frozen Panda T3 source skill bank')
        self.spec,self.policy,self.source_bank,self.feedback=spec,policy,source_bank,feedback
        self.device=torch.device(device);self.mode=int(mode);self.seed=int(seed);self.maximum_query_distance=maximum_query_distance
        self.base=self.BASE_POLICY.load(spec,prior_path,feedback=feedback,modes=[mode],seed=seed)
        self.interceptor=ProtoInterceptor();self.base.pose_transport=self.interceptor
        self.policy.to(self.device).eval();self.frames=[];self.last_tick=-1;self.last_query_distance=None;self.fallback_controls=0;self.irreversible_network_calls=0


class PublicMobileResponseMPCActor(PublicResponseMPCActor,PublicMobileDenseTransportActor):
    """Same five-step endpoint-response optimizer on the explicit mobile base."""


class PublicLivenessMobileDenseTransportActor(PublicMobileDenseTransportActor):
    """Frozen dense policy under the P5 safe-settled public governor."""

    BASE_POLICY=PublicLivenessMatingFeedbackPolicy


class PublicLivenessMobileResponseMPCActor(PublicResponseMPCActor,PublicLivenessMobileDenseTransportActor):
    """Frozen ordinary MPC under the same P5 public governor."""
